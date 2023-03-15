## 注意

若您要在本机上运行这些代码，基于仓库代码作以下修改：
修改 `preprocess.py` 的 `line 125` 为
```python
    return transition_count, kmeans, state_weightes, all_prediction_container
```
修改 `main.py` 的 `line 75` 为
```python
    transition_count, kmeans, state_weightes, all_prediction_container = get_transitions(model, train_dataset, CLUSTER)
```
后在 `main.py` 之后运行即可。

## 计算一个单词对RNN判别的影响力

### 定义
对于Word $w$，考察它的Transition Matrix $E_w$. 如下决定对于类别$X$的影响力：
首先对于KMeans聚类，设每个类别$C$的簇大小(训练集当中，被判定为$C$类别的元素个数)为$K_C$ . 那么，设定权重向量
$$W[i] = \frac{K_{C_i}}{\sum_{j=0}^K K_{C_j}} .$$
定义：Uniform State Distribution 为如下的一个在状态集$S = [K]$上的概率分布：$$\mathrm{Pr}(X = i) = W[i].$$我们记每个状态$s$对应的在类别上的概率分布列为$P_s \in \mathbb{R}^{C}$.其中，$P_s[i]$为状态$s$被分类为类别$C_i$的概率。那么，记单词的的平均影响力定义如下:$$\mathrm{Influence}(w) := \sum_{j=1}^K W[j]P_{s_j}E_w - \sum_{j=1}^K W[j]P_{s_j} \in \mathbb{R}^C.$$那么$\mathrm{Influence}(w)[i]$就是单词$w$在类别$C_i$上的影响力。

### 代码

```python
classid = 0 # 设定classid
# 计算 Influence(w)[classid]的列表。prob_classid 是一个列表，其第 i 个元素是第 i 个单词在类别 classid 上的影响力
kmeans_prediction = kmeans.predict(all_prediction_container)
from collections import Counter
frequencies = Counter(kmeans_prediction)
total = sum(frequencies.values())
for key in frequencies:
    frequencies[key] = frequencies[key] / total 
weight_km = torch.tensor(list(dict(sorted(frequencies.items())).values()))
def out_prob(word):
    mat_word = torch.clone(transition_matrices[word][1:])
    state_w = torch.clone(state_weightes[1:])
    weight_km_c = torch.clone(weight_km).to('cpu')
    mat_word_c = torch.clone(mat_word).to('cpu')
    out_state = torch.matmul(weight_km_c,mat_word_c)[1:]
    for i in range (len(state_w)):
        state_w[i] = state_w[i] * out_state[i]
    return torch.sum(state_w,dim = 0)
    
prob_classid = []
for i in range(len(transition_matrices)):
    prob_classid.append(out_prob(i)[classid])
# TOP-1 WORD ID:
# print(prob_classid.index(max(prob_classid))) 
# -----------------------------------------------------------------
import heapq
# 获取 top-10 影响力单词索引
top_10 = heapq.nlargest(10, range(len(prob_classid)), prob_classid.__getitem__)

import torchtext
# 获取 top-10 影响力单词
for rank in top_10:
    print(torchtext.vocab.Vocab.get_itos(train_dataset)[rank])

```

### 结果

##### News
Sport 分类影响力 top-10:
1. celtics
2. basketball
3. mets
4. cup
5. lockout
6. rangers
7. soccer
8. dodgers
9. knicks
10. lakers

International Politics 分类影响力 top-10:
1. mladic
2. yemen
3. egypt
4. pakistan
5. peru
6. gaddafi
7. syria
8. rebels
9. libyan
10. syrian

#### Toxic
Toxic 分类影响力 top-10:
1. fuck
2. shit
3. moron
4. bitch
5. idiot
6. fucking
7. faggot
8. cunt
9. crap
10. douche

*非* Toxic分类影响力 top-10:
1. mpeg
2. prescott
3. dumba
4. khalij-e-arabi
5. maintainted
6. subtitle
7. cdrtools
8. voegelin
9. signa
10. minling

## 计算基于任务的近义词

### 定义
设训练集的词汇表Vocab为$W$, 其中基于此训练集训练的RNN提取出的词$w \in W$的转移矩阵为$E_w$.那么词$w_0$的**基于任务的近义词**定义为:
$$w_{\mathrm{Nearest}} = \arg \min_{w\in W} \mathrm{Norm}(E_w - E_{w_0}),$$
其中$\mathrm{Norm}(·)$为任意矩阵范数，我们在代码中取定Frobenius范数$||·||_2$.

### 代码
```python
def diff_word(w1,w2):
    w1_m = torch.clone(transition_matrices[w1])
    w2_m = torch.clone(transition_matrices[w2])
    return torch.sum((w1_m - w2_m)**2)
def get_nearest(input_w):
    goal_w = torchtext.vocab.Vocab.get_stoi(train_dataset)[input_w]
    idx = 0
    dist = 1e6
    for word in range(len(transition_matrices)):
        if (diff_word(word,goal_w) < dist) and (word != goal_w):
            idx = word
            dist = diff_word(word,goal_w)
    print(torchtext.vocab.Vocab.get_itos(train_dataset)[idx])
# 使用 get_nearest(word) 就可以打印word的基于任务的近义词了。传入的word是一string。
```

### 一些例子
israel → asia;
happy → stupid;
good → for;
basketball → knicks;
lockout → basketball.

### 另一种定义
第一部分的Uniform State Distribution 的意义为：假定输入分布的数据分布为$\mathcal{D}$. 那么，设定训练集当中不同输入服从这个分布$T\sim \mathcal{D}$.那么，利用Kmeans的簇大小作为概率确定的状态集合上的分布，可以粗略地视为输入服从训练集上的均匀分布(某种程度上，近似于输入文本的数据分布$\mathcal{D}$)的情况下，状态集合上应有的分布。记状态$s_i$对应在输出分类$C_j$上的概率为$s_i(j).$考察此初始分布为$$P_0(s_i) =  \frac{K_{C_i}}{\sum_{j=0}^K K_{C_j}}, P(C_i) = \sum_j P_0(s_j)s_j(i).$$受到单词$w$的影响之后，概率分布变成$$P(s_i) = \sum_k P_0(s_k)E_w(k,i);P_w(C_i) = \sum_j P(s_j)s_j(i).$$记后面这一种在$[C]$上的分布为$D_w$.从而，我们只需要研究两个分布$D_{w_1}$和$D_{w_2}$之间的距离。对于概率分布之间的距离，已经有良好的数学基础：只需要计算KL散度。因而，$w_0$近义词定义为$$w^*_0 = \arg \min_{w\in W} KL(P_{w_0}||P_{w}). $$
### 代码
```python
import scipy
def KL(Y, P):
    return scipy.stats.entropy(Y, P)
def diff_ce(w1,w2):
    w1_m = torch.clone(transition_matrices[w1]).to('cpu')[1:]
    w2_m = torch.clone(transition_matrices[w2]).to('cpu')[1:]
    unif_s = torch.clone(weight_km).to('cpu')
    sD_1 = torch.matmul(unif_s,w1_m)[1:]
    sD_2 = torch.matmul(unif_s,w2_m)[1:]
    state_w1 = torch.clone(state_weightes[1:])
    state_w2 = torch.clone(state_weightes[1:])
    for i in range (len(state_w1)):
        state_w1[i] = state_w1[i] * sD_1[i]
        state_w2[i] = state_w2[i] * sD_2[i]
    P_1 = torch.sum(state_w1,dim = 0)
    P_2 = torch.sum(state_w2,dim = 0)
    return KL(P_1.cpu(),P_2.cpu())
def get_nearest_ce(input_w):
    goal_w = torchtext.vocab.Vocab.get_stoi(train_dataset)[input_w]
    idx = 0
    dist = 1e2
    for word in range(len(transition_matrices)):
        if (diff_ce(goal_w,word) < dist) and (word != goal_w):
            idx = word
            dist = diff_ce(goal_w,word)
    print(torchtext.vocab.Vocab.get_itos(train_dataset)[idx])
#get_nearest_ce('cup') returns worlds
```

### 一些有代表性的结果
lockout -> sports
cup -> worlds
war -> chernobyl

## 其他可能的工作？

### 对抗序列的解释
所谓对抗序列，可以粗略地看成作出很小的扰动就会影响很大的序列。考虑序列$w_1,w_2,\cdots,w_n$, 那么这个序列的**条件数**为(这个名词来源于数值分析)：$$E^* = \prod_{i=1}^n E_{w_i}; \mathrm{Cond}(E^*) = ||(E^*)^{-1}||·||E^*||.$$这里，条件数在对抗性解释的意义是：$$\mathrm{Cond}(E^*) = \max \left\{\left|\frac{E^*x-E^*x^*}{E^*x}\right| /\left|\frac{x-x^{*}}{x}\right|:\left|x-x^{*}\right|<\epsilon\right\}$$可以看到，一定程度上反映了序列对扰动的敏感性。可以找到一些对抗序列和普通序列，计算它们的条件数以表明条件数指标是一个相关的指标。当然也可以直接考虑计算扰动序列当中的转移矩阵作$B_p(\varepsilon)$大小范围内的扰动能够造成的影响。
