# twist：当前实现

## 串联架构

输入 A、B 为同尺寸灰度图，白色 +1、黑色 −1；采样坐标归一化到 [0,1]²。每个配准阶段读取累计变形后的 A 和原始 B，重新切 patch、投影并加入固定二维位置编码。

阶段内堆叠 blocks_per_stage 个双流模块：A 做 SA → FFN，B 做 SA → CA → FFN。CA 中 B 是 Q，更新并解码 B 的查询 token；K/V 来自完成 SA/FFN 的 A。这样每个模块包含两个 SA、一个 CA、两个 FFN，最后一个 A 的 FFN 也参与输出。

每个 B token 投影成二维位移；内部控制点外增加一圈固定零节点。num_stages 个阶段默认参数独立，可通过 share_stage_weights 共享。阶段之间通过当前 A 图像传递几何结果；下一阶段对原始 B 重新编码，B 图像本身不改变。

## 反向采样与连续轨迹

阶段按 t=1 的端点配准依次预测静态位移，保存全部 u_k。查询其他 t 不再运行网络：

$$
g_{k,t}(x)=x+t\,u_k(x),\qquad
G_{0,t}=\mathrm{id},\qquad
G_{k,t}=G_{k-1,t}\circ g_{k,t},\qquad
A_t(x)=A(G_{K,t}(x)).
$$

当前阶段看到的是 A(G_{k-1,1}(x))。每次从原始 A 采样，采用累计坐标；这保留串联依赖并减少重复插值模糊。复合时按相反顺序依次求值各连续小映射，不重新插值粗网格累计映射，也不求逆。头尾对称、时间嵌入和频域算子不纳入首版。

## 固定几何约束

先把边界置零，再计算整个双线性控制场的保守 Lipschitz 上界。令 h_x、h_y 为归一化控制网格间距：

$$
a=\max_{ij}\frac{\|u_{i,j+1}-u_{ij}\|}{h_x},\qquad
b=\max_{ij}\frac{\|u_{i+1,j}-u_{ij}\|}{h_y},\qquad
\widehat L=\sqrt{a^2+b^2}.
$$

按 min(1,ρ/(L̂+ε)) 缩放整个场，ρ=lipschitz_limit<1。边界外为零位移。由此每个连续小映射满足：

$$
\|g_{k,t}(p)-g_{k,t}(q)\|\ge(1-t\rho)\|p-q\|.
$$

不同点不合并，固定边界，允许局部粗细变化；实际栅格仍受采样精度限制。约束通过固定操作执行，梯度能够回传。

## 三类 loss

总比例集中在 config.py 的 LossConfig：

$$
L=w_{\mathrm{mse}}L_{\mathrm{mse}}
+w_{\mathrm{dist}}L_{\mathrm{dist}}
+w_{\mathrm{def}}L_{\mathrm{def}}.
$$

- **软权重 MSE：** 用原始 B 的 m_B=(1−B)/2 作为墨迹权重。每张图分别求软前景和软背景的平均误差，对非空组等权平均，无需调前景倍率。
- **距离场：** 预先计算原始 A 的距离场 D_A，每次在最终累计采样坐标处读取；距离使用归一化坐标单位，不需要中间帧或变形标签。
- **形变正则：** 对各阶段执行约束后的控制网格计算弯曲损失与小位移损失。

$$
L_{\mathrm{dist}}=
\frac{\sum_q m_B(q)D_A(G_{K,1}(q))^2}{\sum_q m_B(q)+\varepsilon},
$$

$$
L_{\mathrm{def}}=
\operatorname{mean}_{k,i,j}
\bigl(\|D_{xx}u_k\|^2+2\|D_{xy}u_k\|^2+\|D_{yy}u_k\|^2\bigr)
+\eta\,\operatorname{mean}_{k,i,j}\|u_k\|^2.
$$

差分按 h_x、h_y 缩放，η=displacement_ratio。loss 只针对最终端点及各小位移，不单独监督每阶段图像。反方向训练可交换 A、B，并缓存 B 的距离场；两次预测不要求互为逆。

当前实现包括模型、约束、距离场准备、loss、测试、参数统计、单字实验与全数据集同字配对训练；通用数据生成及外观修正分支留待后续。
