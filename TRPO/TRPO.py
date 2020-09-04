import tensorflow as tf
import numpy as np
from tensorflow.train import AdamOptimizer
from scipy import signal
from gym.spaces import Box, Discrete
import os
import gym

# 定义除数修正系数
EPS = 1e-8


class GAEBuffer(object):
    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        """
        :param obs_dim: 观测状态的维度
        :param act_dim: 行动的维度
        :param size: 每一个Epoch训练使用的样本个数，缓存器的最大容量
        :param info_shapes: 计算KL散度的必要参数的维度
        :param gamma: 远期奖励的折扣比率
        :param lam: 广义优势估计的额外折扣比率
        """
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        if act_dim > 0:
            self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        else:
            self.act_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32)
                          for k, v in info_shapes.items()}
        self.sorted_info_keys = self.info_bufs.keys()
        self.gamma, self.lam = gamma, lam
        # ptr是插入数据点时的位置索引，path_start_idx是一个完整决策回合的起始位置
        # max_size是一个完整学习回合的缓存器最大尺寸
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        """
        每一次决策和反馈完成之后，就需要将该轮次的观测，行动，奖励，对观测的估值，采取该行动的对数似然，
        以及基于观测所给出的可行行动空间分布参数存储到相应的位置当中
        """
        # 插入数据时要确保插入索引小于缓存器
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        """
        当回合终止的时候，需要根据完整的决策过程，计算出该回合的回报估计序列和广义优势序列
        两种情况，一，如果是缓存器已满，那么就是截断了某一个回合，那么就需要根据当时的观测
        做出一个观测估值，填入到回合数据的尾部；二，如果是回合正常终止，那么最后一个状态的
        价值就是0。因为后续没有奖励了。
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # 这里定义出来的优势是广义优势，因为策略在一个决策回合中保持不变，所以在某一个观测点上某一个
        # 行动所带来的影响会产生明显的连锁反应，后续的优势能够产生也是由于当前行动所决定的。所以广义
        # 优势就需要将后期优势也累加到当前优势当中，只不过折扣力度要比回报高一些。
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma*self.lam)
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        # 只有在进行学习的时候才需要在可以从缓存器中抽取数据，抽取数据之后，缓存器就清空
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # 由于目前的估值网络不准确，可能导致优势整体上偏高或者偏低，所以需要移除均值和方差
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean)/adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf,
                ] + [self.info_bufs[k] for k in self.sorted_info_keys]


def placeholder_generator(shape, dtype_str, name):
    if dtype_str == 'float':
        dtype = tf.float32
    elif dtype_str == 'int':
        dtype = tf.int32
    return tf.placeholder(dtype=dtype, shape=shape, name=name)


def mlp(x, hidden_sizes=(32, ), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def categorical_kl(logp0, logp1):
    all_kls = tf.reduce_sum(tf.exp(logp1)*(logp1-logp0), axis=1)
    return tf.reduce_mean(all_kls)


def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    # 计算出状态经策略网络映射之后的(未归一化)对数概率向量
    logits = mlp(x, list(hidden_sizes) + [act_dim], activation, output_activation)
    # 归一化对数概率向量
    logp_all = tf.nn.log_softmax(logits)
    # 根据计算出来的分布参数，生成一个行动决策
    pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1, name="Action")
    # 计算出某一个给定行动的对数似然
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    # 计算出当前随机决策行动的对数似然
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    # 前置策略的归一化对数似然占位符
    old_logp_all = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))
    # 计算当前策略和前置策略的KL散度
    d_kl = categorical_kl(logp_all, old_logp_all)

    info_new = {'logp_all':logp_all}
    info_old_ph = {'logp_all':old_logp_all}
    # 返回值依次为：
    #  （依赖于状态观测的）行动决策点计算节点
    #  （依赖于状态观测和行动的）对数似然计算节点
    #  （依赖于状态观测的）当前随机决策的对数似然计算节点
    #  （依赖于状态观测的）策略网络信息度量计算节点
    #  前置策略网络信息度量的占位符
    #  （依赖于状态观测的）前置策略网络和当前策略网络的KL散度计算节点
    return pi, logp, logp_pi, info_new, info_old_ph, d_kl


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5 * ( ( (mu1-mu0)**2 + var0) / (var1+EPS) - 1) + log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    # 计算出观测状态经策略网络映射之后均值向量
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    # 创建一个待估参数向量，代表标准差向量
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32)
                              , dtype=tf.float32)
    std = tf.exp(log_std)
    # 根据高斯分布的均值和方差计算出随机决策点
    pi = tf.add(mu, tf.random_normal(tf.shape(mu)) * std, name='Action')
    # 根据对数概率密度函数计算某一个行动在当前状态下的对数似然
    logp = gaussian_likelihood(a, mu, log_std)
    # 根据对数概率密度函数计算当前随机决策点的对数似然
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    # 创建前置策略网络的信息度量的占位符
    old_mu_ph = placeholder_generator((None, act_dim), 'float', 'old_mu_PH')
    old_log_std_ph = placeholder_generator((None, act_dim), 'float', 'old_log_std_ph')
    # 根据前置网络信息度量参数和当前网络的信息度量参数计算
    d_kl = diagonal_gaussian_kl(mu, log_std, old_mu_ph, old_log_std_ph)
    info_new = {'mu':mu, 'log_std':log_std}
    info_old_ph = {'mu':old_mu_ph, 'log_std':old_log_std_ph}
    # 返回值依次为：
    #  （依赖于状态观测的）行动决策点计算节点
    #  （依赖于状态观测和行动的）对数似然计算节点
    #  （依赖于状态观测的）当前随机决策的对数似然计算节点
    #  （依赖于状态观测的）策略网络信息度量计算节点
    #  前置策略网络信息度量的占位符
    #  （依赖于状态观测的）前置策略网络和当前策略网络的KL散度计算节点
    return pi, logp, logp_pi, info_new, info_old_ph, d_kl


def mlp_actor_critic(x, a, hidden_sizes=(64, 64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):
    # 如果没有指定策略网络的具体对象
    if policy == None:
        # 如果行动空间是连续的，那么就将策略网络定义为全连接高斯分布网络
        if isinstance(action_space, Box):
            policy = mlp_gaussian_policy
        # 如果行动空间是离散的，那么就将策略网络定义为全连接离散分布网络
        elif isinstance(action_space, Discrete):
            policy = mlp_categorical_policy
    # 声明一个变量空间Pi，包含策略网络参数集合
    with tf.variable_scope('pi'):
        # 返回值依次为：
        #  （依赖于状态观测的）行动决策点计算节点
        #  （依赖于状态观测和行动的）对数似然计算节点
        #  （依赖于状态观测的）当前随机决策的对数似然计算节点
        #  （依赖于状态观测的）策略网络信息度量计算节点
        #  前置策略网络信息度量的占位符
        #  （依赖于状态观测的）前置策略网络和当前策略网络的KL散度计算节点
        policy_outs = policy(x, a, hidden_sizes, activation, output_activation, action_space)
        pi, logp, logp_pi, info_new, info_old_ph, d_kl = policy_outs
    # 声明一个变量空间V，代表估值网络参数集合
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes) + [1], activation, None), axis=1)

    return pi, logp, logp_pi, info_new, info_old_ph, d_kl, v


def hessian_vector_product(f, params):
    gm = tf.gradients(f, params)
    g = tf.concat([tf.reshape(x, (-1,)) for x in gm], axis=0)
    x = tf.placeholder(tf.float32, shape=g.shape)
    fxxs = tf.gradients(tf.reduce_sum(g * x), params)
    fxx = tf.concat([tf.reshape(i, (-1,)) for i in fxxs], axis=0)
    return x, fxx


def trpo(env_fn, actor_critic, save_path, exp_name, env_obs_dim, env_obs_type,
         env_action_dim, env_action_type, restartTrain=False, modelFile='',
         step_per_learn_epoch=4000, damping_coef=0.1, ac_kwargs=dict(),
         learn_epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3, train_v_iters=80,
         cg_iters=10, backtrack_iters=10, backtrack_coef=0.8, lam=0.97, max_ep_len=1000,
         logger_kwargs=dict(), save_num=5, algo='trpo', seed=0,):
    """
        env_fn:函数，无参数，调用该函数可以直接创建出智能体与之交互的环境对象
        actor_critic:智能体对象生成器，构造参数可以由ac_kwargs来传递，所构造的对象包括了策略网络与估值网络
        save_path:对象文件存储路径字符串，可以将训练好的TF模型和参数文件存储在该路径下
        ac_kwargs:智能体对象生成参数字典
        seed:全局的随机种子
        step_per_learn_epoch:每一个学习回合中的总步数
        learn_epochs:总学习回合数
        gamma:奖励的折扣系数
        delta:KL散度的上界
        vf_lr:估值网络学习率
        train_v_iters:每一个学习回合中对估值网络的学习次数
        damping_coef:数值计算的稳健参数，避免Hessian矩阵为退化阵。
        cg_iters:进行共轭梯度计算时计算共轭分量的数量
        backtrack_iters:合法值线搜索的迭代次数
        backtrack_coef:线搜索的缩减比率
        lam:广义优势估计的附加缩减比率
        max_ep_len:单一决策回合的最大决策轮次，设置该值以避免某一些回合陷入无法停止的局面
        logger_kwargs:日志对象的必要构造参数
        save_freq:模型的缓存频率
        algo:TRPO算法和NPG算法的唯一区别就在于是否进行线搜索来确保约束条件成立，该参数可以
                管理是否启用线搜索

    """
    # 设置运算过程的随机种子
    if seed != 0:
        tf.set_random_seed(seed)
        np.random.seed(seed)
    # 生成策略运行的模拟环境
    env = env_fn()
    # 确定观测空间维度
    obs_dim = env_obs_dim
    # 确定行动空间维度
    act_dim = env_action_dim
    # 向ac_kwargs中添加行动空间的信息，以方便策略网络进行使用
    ac_kwargs['action_space'] = env.action_space
    # +++++++++++++++++ 定义全部的占位符和张量（操作节点） ++++++++++++++
    # 定义状态观测的占位符
    x_ph_dtype = env_obs_type
    x_ph_shape = (None, obs_dim)
    x_ph_name = 'obs_PH'
    x_ph = placeholder_generator(x_ph_shape, x_ph_dtype, x_ph_name)
    # 定义行动的占位符
    a_ph_dtype = env_action_type
    if act_dim == 0:
        a_ph_shape = (None, )
    else:
        a_ph_shape = (None, act_dim)
    a_ph_name = 'act_PH'
    a_ph = placeholder_generator(a_ph_shape, a_ph_dtype, a_ph_name)
    # 广义优势估计占位符
    adv_ph = placeholder_generator((None, ), 'float', 'adv_PH')
    # 回报估计占位符
    ret_ph = placeholder_generator((None, ), 'float', 'ret_PH')
    # 前置对数似然占位符
    old_logp_ph = placeholder_generator((None, ), 'float', 'oldLogp_PH')
    # 返回值依次为：
    #  （依赖于状态观测的）行动决策点计算节点
    #  （依赖于状态观测和行动的）对数似然计算节点
    #  （依赖于状态观测的）当前随机决策的对数似然计算节点
    #  （依赖于状态观测的）策略网络信息度量计算节点
    #  前置策略网络信息度量的占位符
    #  （依赖于状态观测的）前置策略网络和当前策略网络的KL散度计算节点
    pi, logp, logp_pi, info, info_phs, d_kl, v = actor_critic(x_ph, a_ph, **ac_kwargs)
    # 收集全部的占位符，包括：
    # 状态观测占位符， 行动占位符， 长期优势占位符， 长期回报占位符，
    # 前置策略网络行动对数似然占位符， 前置策略网络信息张量占位符
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, old_logp_ph] + list(info_phs.values())
    # 决策计算节点集合，包括（依赖于状态观测的）随机决策点计算节点，估值计算节点，
    # 随机决策点对数似然计算节点，策略网络信息张量计算节点
    get_action_ops = [pi, v, logp_pi] + list(info.values())
    # 每一个学习更新回合最大的决策次数
    step_per_learn_epoch = int(step_per_learn_epoch)
    # 映射info里面存放了用于计算KL散度的策略网络的某些结果张量
    # 而映射info_phs里面存放了用于计算KL散度的前置策略网络相关结果的占位符
    # 这里汇总了各个相关结果的形状结果
    info_shapes = {k:v.shape.as_list()[1:] for k, v in info_phs.items()}
    # 声明一个历史数据缓存器，
    # gamma是奖励折扣系数，lam是优势超额折扣系数，这两个参数都应该取值在0到1之间，且很靠近1
    buf = GAEBuffer(obs_dim, act_dim, step_per_learn_epoch, info_shapes, gamma, lam)
    # 此处要计算重要性采样比率计算节点， 即给定前置状态观测和前置策略网络给出的行动，
    # 计算新的策略网络相应的对数似然。当然新的策略网络目前还不存在，
    # 所以本质上是一个以策略网络权重为参数的重要性采用比率函数
    ratio = tf.exp(logp - old_logp_ph)
    # 在这个比率的基础上继续做运算，乘以广义优势估计占位符， 取平均数，取负数，
    # 这样就可以构建出目标损失函数，对其求最小值，即可对策略网络的参数进行优化
    pi_loss = -tf.reduce_mean(ratio * adv_ph)
    # 利用每一个控制回合结束后计算出来的长期回报占位符计算估值网络v的偏差损失，
    # 并用该损失修正估值网络参数
    v_loss = tf.reduce_mean((ret_ph - v)**2)
    # 在估值网络的损失上定义一个优化器
    train_v_f = AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)
    # 获取策略网络全部的待估参数
    scope = 'pi'
    pi_params = [x for x in tf.trainable_variables() if scope in x.name]
    # 使用策略网络损失对所有的可训练参数进行求导
    gradient = tf.gradients(xs=pi_params, ys=pi_loss)
    gradient = tf.concat([tf.reshape(i, (-1, )) for i in gradient], axis=0)
    # 由于KL散度计算节点依赖于策略网络的权值矩阵和前置策略网络的信息张量占位符，
    # 所以根据数学推导，KL散度函数的Hessian矩阵与任何向量x的乘积不需要显式计算
    # Hessian矩阵，而由这里提供的一个张量HVP来负责处理，填充一个占位符之后，hvp计算节点
    # 的结果就是该向量与KL散度Hessian矩阵的计算结果
    v_ph, hvp = hessian_vector_product(d_kl, pi_params)
    # damping_coeff是一个稳健系数，主要为了避免Hessian矩阵为退化阵
    if damping_coef > 0:
        hvp += damping_coef * v_ph
    # 获取策略网络的参数矩阵
    get_pi_params = tf.concat([tf.reshape(i, (-1, )) for i in pi_params], axis=0)
    # v_ph是提供权限策略网络参数的占位符，为一个行向量，（这里是和上面的计算共享了占位符，因为两者形状相同）
    # set_pi_params计算节点对应着将策略网络参数更新到新参数上
    flat_size = lambda p: int(np.prod(p.shape.as_list()))
    splits = tf.split(v_ph, [flat_size(p) for p in pi_params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(pi_params, splits)]
    set_pi_params = tf.group([tf.assign(p, p_new) for p, p_new in zip(pi_params, new_params)])
    # +++++++++++++++++ 定义全部的占位符和张量（操作节点） ++++++++++++++
    # 创建计算资源会话
    sess = tf.Session()
    if restartTrain:
        modelSaver = tf.train.Saver()
        modelSaver.restore(sess,
                           tf.train.latest_checkpoint(os.path.join(save_path, exp_name))
                           )
    else:
        # 完成全局变量的初始化操作
        sess.run(tf.global_variables_initializer())
    #

    def cg(Ax, b, cg_iters=cg_iters):
        # 定义共轭梯度算法， 求解在特定策略网络参数矩阵的条件下
        # （未知KL散度的Hessian矩阵和已知广义优势损失的雅克比向量）调整二阶泰勒展开最优点的位置。
        # 该方法的优点是不需要给出Hessian矩阵和它的逆
        x = np.zeros_like(b)
        # 总残差向量
        r = b.copy()
        # 共轭分量（初始的共轭分量就是总残差向量）
        d = r.copy()
        # 总残差向量的模
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            # 当前的共轭分量在Hessian矩阵投影空间中的向量
            z = Ax(d)
            # 在投影空间中的向量与原空间中的向量的乘积可以解释多大比率的总残差
            alpha = r_dot_old / (np.dot(d, z) + EPS)
            # 按照这一比率将该共轭分量混入到目标结果向量中
            x += alpha * d
            # 从总残差向量中扣除已经得到解释的部分
            r -= alpha * z
            # 计算出新的残差向量的模
            r_dot_new = np.dot(r, r)
            # 在总残差向量的基础上调整得到下一个共轭分量，调整的方向是向当前的共轭分量方向调整
            # 调整的力度是剩余残差的比率
            d = r + (r_dot_new / r_dot_old) * d
            r_dot_old = r_dot_new
        return x

    def update():
        # 将全部的占位符和对应的缓存数据组织起来
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        # 供共轭梯度方法使用的一个计算函数， 可以快速计算Hessian矩阵和任意向量x之间的乘积
        Hx = lambda x: sess.run(hvp, feed_dict={v_ph: x, **inputs})
        # 分别计算出负策略优势损失对参数向量的导数， 负策略优势损失，值函数估计损失
        g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
        # 利用共轭梯度方法直接求解出来当前策略的二阶（泰勒展开）最大优势点
        x = cg(Hx, g)
        # 计算出更新权重
        alpha = np.sqrt(2 * delta / (np.dot(x, Hx(x)) + EPS))
        # 获取当前的策略网络参数
        old_params = sess.run(get_pi_params)

        def set_and_eval(step):
            # 根据更新公式设置新的策略网络参数
            sess.run(set_pi_params, feed_dict={v_ph:old_params - alpha * step * x})
            return sess.run([d_kl, pi_loss], feed_dict=inputs)

        if algo == 'npg':
            # 如果使用的算法是NPG，那么就直接将更新步长设置为1
            kl, pi_l_new = set_and_eval(step=1.0)
        elif algo == 'trpo':
            # 如果使用的算法是TRPO，那么就需要使用线搜索来确定合法的更新步长
            for j in range(backtrack_iters):
                # 测试按比例收缩更新之后的KL散度和策略优势是否满足限制条件
                kl, pi_l_new = set_and_eval(step=backtrack_coef ** j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    print('  [LinearSearch]: Accepting New Params At Step %d of Line Search.'%j)
                    break
                else:
                    # 如果线搜索到最后一步，仍然不能满足KL散度和策略优势的要求，
                    # 那么就放弃此方向上的更新
                    print('  [Warning]: Line Search Failed! Keeping Old Params.')
                    kl, pi_l_new = set_and_eval(step = 0)
        # 对估值网络进行训练
        for _ in range(train_v_iters):
            sess.run(train_v_f, feed_dict=inputs)
        # 根据新的估值网络计算出新的估值损失
        v_l_new = sess.run(v_loss, feed_dict=inputs)
    # 生成需要缓存模型的索引
    dirPath = os.path.join(save_path, envName)
    if os.path.exists(dirPath):
        pass
    else:
        os.mkdir(dirPath)
    saveIndex = np.arange(save_num)[2:]*learn_epochs//save_num
    modelSaver = tf.train.Saver()
    # 生成计算流程图
    writer = tf.summary.FileWriter('./ComputeGraph', sess.graph)
    writer.flush()
    writer.close()
    # 重置环境，设置总奖励记录值和总决策次数记录值
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(learn_epochs):
        print('++++ Excuting [%d] Epoch ++++'%(epoch+1))
        for t in range(step_per_learn_epoch):
            # 基于当前的状态观测，计算出策略网络给出的行动Pi
            # 估值网络对于当前状态观测的估值v_t，策略网络对给出行动的对数似然logp_t，
            # 以及之后用于计算KL散度的必要参数
            agent_outs = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1, -1)})
            a_t, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1],\
                                       agent_outs[2], agent_outs[3:]
            # 环境执行这一个决策，给出衍生的状态观测， 奖励， 结束信号
            o_new, r_t, d_t, _ = env.step(a_t)
            # 累加总奖励和总决策计数
            ep_ret += r_t
            ep_len += 1
            # 向缓存器中记录当前回合的状态观测，行动，奖励，估值，对数似然，和KL散度计算参数
            buf.store(o, a_t, r_t, v_t, logp_t, info_t)
            # 更新状态观测到最新的状态观测
            o = o_new
            terminal = d_t or (ep_len == max_ep_len)

            if terminal or (t == step_per_learn_epoch-1):
                if not terminal:
                    print('  [Warning]: Trajectory Cut Off by Epoch at %d steps.'%ep_len)
                last_val = 0 if d_t else sess.run(v, feed_dict={x_ph: o.reshape(1, -1)})
                buf.finish_path(last_val)
                if terminal:
                    pass
                print('  [TotalReward]: %d'%ep_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0
        # 完成一次模拟实验，对策略网络和估值网络进行更新
        update()
        if epoch in saveIndex:
            modelSaver.save(sess, os.path.join(save_path, exp_name, '%s_model.ckpt'%algo),
                            global_step=epoch)

    print('  [Success]: Experiment Over')
    modelSaver.save(sess, os.path.join(save_path, exp_name, '%s_model.ckpt' % algo),
                    global_step=epoch)
    sess.close()


def env_fn():
    env = gym.make(envName)
    env = env.unwrapped
    return env


if __name__ == '__main__':
    #envName = 'MountainCar-v0'
    #envName = 'Acrobot-v1'
    #envName = 'CartPole-v1'
    #envName = 'MountainCarContinuous-v0'
    #envName = 'Pendulum-v0'
    #envName = 'BipedalWalker-v3'
    envName = 'BipedalWalkerHardcore-v3'

    controlType = 'train'

    if envName == 'MountainCar-v0':
        max_ep_len = 2000
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 2, 0
        env_obs_type, env_act_type = 'float', 'int'
        learn_epochs = 200
        gamma = 0.999; lam = 0.99
        step_per_learn_epoch = 5000
        delta = 0.02
        cg_iters = 10
        restartTrain = False

    if envName == 'Acrobot-v1':
        max_ep_len = 2000
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 6, 0
        env_obs_type, env_act_type = 'float', 'int'
        learn_epochs = 200
        gamma = 0.999
        lam = 0.99
        step_per_learn_epoch = 5000
        delta = 0.01
        cg_iters = 10
        restartTrain = False

    if envName == 'CartPole-v1':
        max_ep_len = 1500
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 4, 0
        env_obs_type, env_act_type = 'float', 'int'
        learn_epochs = 200
        gamma = 0.999
        lam = 0.99
        step_per_learn_epoch = 5000
        delta = 0.01
        cg_iters = 10
        restartTrain = False

    if envName == 'MountainCarContinuous-v0':
        max_ep_len = 1500
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 2, 1
        env_obs_type, env_act_type = 'float', 'float'
        learn_epochs = 200
        gamma = 0.999
        lam = 0.99
        step_per_learn_epoch = 5000
        delta = 0.01
        cg_iters = 10
        restartTrain = False

    if envName == 'Pendulum-v0':
        max_ep_len = 1500
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 3, 1
        env_obs_type, env_act_type = 'float', 'float'
        learn_epochs = 200
        gamma = 0.95
        lam = 0.92
        step_per_learn_epoch = 5000
        delta = 0.01
        cg_iters = 10
        restartTrain = False

    if envName == 'BipedalWalker-v3':
        max_ep_len = 1500
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 24, 4
        env_obs_type, env_act_type = 'float', 'float'
        learn_epochs = 200
        gamma = 0.95
        lam = 0.95
        step_per_learn_epoch = 5000
        delta = 0.015
        cg_iters = 20
        restartTrain = True

    if envName == 'BipedalWalkerHardcore-v3':
        max_ep_len = 1500
        savePath = './trainedModel'
        env_obs_dim, env_act_dim = 24, 4
        env_obs_type, env_act_type = 'float', 'float'
        learn_epochs = 400
        gamma = 0.95
        lam = 0.95
        step_per_learn_epoch = 5000
        delta = 0.015
        cg_iters = 20
        restartTrain = False


    if controlType == 'train':
        trpo(env_fn, mlp_actor_critic, savePath, envName, env_obs_dim,
         env_obs_type, env_act_dim, env_act_type, max_ep_len=max_ep_len, learn_epochs=learn_epochs,
         gamma=gamma, lam=lam, step_per_learn_epoch=step_per_learn_epoch, delta=delta,
         restartTrain=restartTrain, cg_iters=cg_iters)
    if controlType == 'test':
        # 测试模型
        import time
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(savePath, envName, 'trpo_model.ckpt-199.meta'))
        module_file = tf.train.latest_checkpoint(os.path.join(savePath, envName))
        saver.restore(sess, module_file)
        computeGraph = tf.get_default_graph()
        x_ph = computeGraph.get_tensor_by_name('obs_PH:0')
        pi = computeGraph.get_tensor_by_name('pi/Action:0')
        env = env_fn()
        for epoch in range(10):
            print('Start %d Epoch'%epoch)
            o = env.reset()
            for _ in range(2000):
                env.render()
                time.sleep(0.02)
                a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
                o, r, d, _ = env.step(a)
                if d:
                    break
        env.close()
        sess.close()







