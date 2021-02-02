智源-水利知识图谱构建挑战赛

- 链接：http://competition.baai.ac.cn/c/37/format/introduce
- 最终名次：12 / 328
- 团队成员：@ymcdull @zhengheng

代码拷贝自 https://github.com/lonePatient/BERT-NER-Pytorch

进行了以下修改：

- 增加了 bien 数据集
- CRF 添加 adv 参数 (FGM)

需要将赛题数据放入 raw_data 目录下, 将 bert-base-chinese 预训练模型放入 prev_trained_model/bert-base-chinese 里. 本赛题的脚本放在了 biendata 目录下, 按顺序依次执行脚本即可.

分数表现:


|                                     |   榜上分数  | 
|-------------------------------------|-------------|
|BERT+CRF                             |0.835        |
|BERT+SPAN                            |0.843        |
|BERT+CRF+FGM                         |0.838        |
|BERT+SPAN+FGM                        |0.848        |
|BERT+SPAN+FGM (hyperparamters blend) |0.854        |

后处理:

1. SPAN 会出现比较长的实体结果, 原因是有两个连续的同样类别的实体, 需要根据情况剔除或者是切割出来;
2. 部分 TER 可以采用训练集字典来直接匹配, 但上分非常困难, 因为有明显漏标的情况.

后处理后榜上分数为 0.86044

PS:

1. 赛题数据不是太好, 很容易过拟合, 而且感觉测试集中漏标的很多(特别是 TER 类的实体);
2. 尝试了多个预训练模型, 反而是 base 表现最好; 
3. epochs 不能过多, 容易过拟合;
4. 多模型融合先投票然后再取交集的方式, 比取并集的效果要好(以上四点好像都是因为测试集漏标太多导致的)
