项目诊断与研究建议｜2026-10-24

本报告基于当前工作区的源码、配置、已有 JSON 结果和截至 2026-10-24 检索到的一手文献。重新运行了现有测试和小规模诊断，没有重新训练模型，也没有运行 VASP。既有实验数字与本次诊断分开说明。

**总体判断：值得继续，但应先修复周期几何，再把研究问题收敛到“紧凑、可部署的周期体系 SCF 初始化”。** 现有结果已经证明环境相关密度可以学习，而且场重构较快；尚未证明实际 SCF 净加速。单凭“等变 GNN 输出原子中心系数、重构连续密度”很难形成充分的新颖性。

**已有结果说明了什么**

| 结果 | 当前证据 | 可以支持的结论 |
|---|---|---|
| QM9 500 小分子，360/40/100 划分 | MACE-style 密度监督的测试 NMAE 约 3.107%；早期模型约 3.911% | 在小数据、固定 probe 设置下，密度监督有效 |
| NMC 官方 1450/50/500 划分 | 500 个完整网格，逐结构平均 NMAE 1.14740%，全局汇总 1.14558% | 已有完整网格的周期材料基准 |
| POTCAR 原子叠加参考 | 500 个结构平均 NMAE 6.99020% | 相对此脚本重建的 SAD，模型密度误差降低约 83.6% |
| ChargE3Net 本地复核 | 100 个结构、各 1000 probes，平均 0.06048%；sample 1 全网格 0.04708% | 本地复核支持其高精度，但不是完整 500 个结构的重新评测 |
| 固定系数空间的逐结构拟合 | 3 个结构 full-grid L1 数值拟合平均 0.67051% | 表示与优化值得优先调查；不是已证明的表示误差下限 |
| 当前 NMC 速度 | 系数预测均值 0.03176 s，重构均值 0.46592 s | 约 0.498 s/结构；约 93.6% 时间花在重构 |
| SCF | 10 个结构输入已经准备；本地预检结果没有收敛配对，10 结构目录未找到已完成 OUTCAR/OSZICAR | 目前不能报告 SCF 步数或墙钟加速 |

数据依据：[NMC 完整网格结果](<C:/Users/User/Desktop/density EQGNN/runs/nmc_full_l2_to_l4_mace/test_full_grid_epoch176.json>)、[POTCAR SAD 结果](<C:/Users/User/Desktop/density EQGNN/runs/nmc_vasp_sad/metrics.json>)、[三结构拟合](<C:/Users/User/Desktop/density EQGNN/runs/nmc_full_l2_to_l4_mace/basis_oracle_l1_test3.json>)、[QM9 实验报告](<C:/Users/User/Desktop/density EQGNN/reports/qm9_coefficient_gnn_l3r12.md>)。

POTCAR SAD 数字来自仓库脚本对原子 form factor 的重建，不能仅凭文件中的 baseline 名称就视为已与 VASP 原生第零步逐点核验。正式 SCF 对照应使用真正的 ICHARG=2。QM9 的 3.107% 来自小数据、采样点测试，也不应直接与完整 QM9 文献结果作同条件排名。

**首先要解决的实现问题**

1. **原子图和密度 decoder 都没有实现完整周期像。** 两处都把分数位移逐分量 round，然后只保留一个像。配置中原子图 cutoff=6 Å，密度支撑 cutoff=3 Å。对 NMC sample 1，最短枚举晶格平移约 5.55016 Å，这两个半径都不能依靠单像实现完整邻域。原子图还直接去掉全部 i=j，因而遗漏非零晶格平移的自身像。

   本次用真实 sample 1 的 POSCAR 枚举整数平移 [-3,3]^3：当前 6 Å 图有 1362 条有向边，完整周期邻域有 4016 条；274 对有序原子的逐分量 round 位移并非真正最短像。这还说明：对斜晶胞，“分数坐标 wrap”与“求最近像”不是同一件事。

   独立反例：5 Å 立方胞、单原子、3 Å 支撑，在胞中面重构一个宽的球形基函数，当前实现只算出显式周期求和的一半。5 Å 单原子胞、6 Å 原子图，应有 6 个最近自身周期像，当前返回 0。

   本次还固定 sample 1 的已训练系数，将结构和系数复制成 2×2×1 超胞，在相同 2048 个空间点比较密度。只改变胞表示，密度相对 L1 差异就有 **0.07606%**。这是表示一致性误差，不是对 DFT 的 NMAE，也不能直接从已有 1.1474% 中扣除。

   修复应同时覆盖原子图、decoder 和 oracle 拟合器：显式保留 cutoff 内全部平移像；只排除 i=j 且平移为零的边。补充斜晶胞、单原子晶胞、原胞/超胞一致性、跨边界连续性及旋转协变检查。修复后重新拟合和训练，旧 checkpoint 不应直接当作新算子的正式结果。

   定位：[decoder](<C:/Users/User/Desktop/density EQGNN/src/density_eqgnn/compact_basis.py:54>)、[原子图](<C:/Users/User/Desktop/density EQGNN/src/density_eqgnn/models/mace_like_model.py:311>)、[NMC 配置](<C:/Users/User/Desktop/density EQGNN/configs/nmc_full_l2_to_l4_mace.yaml>)。

2. **角向基函数没有保证原子中心的正则性。** 当前所有 l 共用 exp(-r²/2σ²)，乘单位方向的 Y_lm，没有 r^l 或其他随 l 消失的因子。对 l>0 的任意非零系数，表示允许在原子中心出现方向相关极限。诊断中，一个合法 l=1 系数在 x=±10^-7 Å 给出约 ±0.48860，而两点距离仅 2×10^-7 Å。网络可能学出部分抵消，但架构没有保证。

   建议比较规范化的 r^l exp(-αr²)Y_lm、solid harmonics 或其他满足原点正则性的 l 相关径向基。径向尺度范围应与 support cutoff 解耦；现在扩大 cutoff 会同时扩大最小 Gaussian 宽度。不同元素、不同 l 使用不同尺度范围也值得做 oracle 消融。以上属于需要实验证实收益的表示改进，并不保证单独降低测试误差。

   定位：[径向尺度](<C:/Users/User/Desktop/density EQGNN/src/density_eqgnn/compact_basis.py:28>)、[基函数构造](<C:/Users/User/Desktop/density EQGNN/src/density_eqgnn/compact_basis.py:63>)。

3. **紧凑系数路径与旧接口未完全统一。** 数据增强仍按旧 scalar/vector/3×3 tensor 布局旋转系数，无法处理 radial-major 的球谐系数。构造 12 radial、l=3、192 维系数并开启增强，会触发 reshape 异常。当前 QM9 系数配置关闭了增强，NMC density-only 没有系数标签，因此这个问题不能用来解释已报告的模型失败；它是未来复用时会触发的缺陷。应明确标注 coefficient layout，并使用每个 l 的 Wigner-D 变换。

   通用预测脚本也固定加载旧 EquivariantCoefficientGNN，不能直接加载当前 mace_like 配置。需要统一模型 factory、basis、训练与推理入口。

   定位：[数据增强](<C:/Users/User/Desktop/density EQGNN/src/density_eqgnn/data/dataset.py:78>)、[通用推理入口](<C:/Users/User/Desktop/density EQGNN/scripts/predict_density.py:39>)。

现有测试本次为 **11 passed，13.90 s**。这些测试证明已有测试范围内的行为，并未覆盖上述周期像和 compact 系数增强反例。本次诊断数字保存于 [diagnostics.json](<C:/Users/User/Desktop/density EQGNN/runs/research_review_20261024/diagnostics.json>)。

**需要校准的研究结论**

- **0.6705% 不是固定基组的数学下限。** 固定位置与基函数后，L1 系数拟合是凸优化，但目前有限步 Adam/smooth-L1 L-BFGS 的可行解只给最优误差的上界。再加上周期算子本身有问题，不能据此断言“GNN 再好也只能到 0.67%”。应在修正算子上扩大拟合样本，检查精度、收敛和多初始化；必要时用原始-对偶间隙评估求解充分性。比较相同结构、相同网格、相同损失下的 GNN 和数值拟合。
- **系数 MSE 失败不等于系数学习不可行。** 本项目的强重叠固定 Gaussian 拟合标签病态，不能推广成整个原子基路线的否定结论。继续以密度监督为主是合理的；如果要引入系数辅助监督，可比较带 overlap 矩阵的 ΔcᵀSΔc，或 Coulomb metric，而不是独立逐系数归一化 MSE。S 是同一密度基的重叠矩阵，这种损失才反映对应的场误差。
- **NMC 是材料家族内的基准，不代表一般周期材料外推。** 仓库已有分析称 71.6% 测试结构的化学式出现在训练集；这里未独立重算该比例。同化学式重叠本身不是数据泄漏，也不使官方随机划分无效，但不能支持跨化学组成泛化。建议另设 Li 含量区间、Ni/Mn/Co 组成、结构簇及不同超胞大小的隔离测试。
- **不要把低阶 trunk 到高阶 readout 当作充分的新颖性。** 当前实现是 MACE-style tensor-product message passing，并非原版 MACE 的完整高体阶架构。若比较真正 MACE，需要明确实现和预算差异。l_hidden=2、readout edge l=2 只能到 l_density=4；扫描 l_density=6 时，必须同步调整 readout 的耦合能力。

**训练和效率的优先事项**

当前每结构保存 16384/439040≈3.73% 网格，训练再从其中抽 2048 点。随机旋转不增加这些点相对于原子的采样覆盖。应缓存完整密度网格，在训练时动态采样；可混合均匀点、近核点、成键区点，若采用重要性采样，要保留相应权重，避免无意改变评估目标。验证和测试保持确定的完整网格或独立固定点。

history.json 中 176 个 epoch 的学习率均为 0.001，ReduceLROnPlateau 尚未产生衰减。可以在固定训练预算下比较 cosine/power decay，但不应把“没有降学习率”直接当作现有误差的已证实原因。至少报告 3 个随机种子及同预算消融。

现有实现先计算所有原子对再 mask，decoder 也构造稠密 atom×probe 张量，因此当前代码不能声称大体系线性扩展。应引入稀疏周期邻居和局部 scatter 重构。尤其约 93.6% 的本地预测时间在 decoder，优化重构的收益可能比继续压缩 trunk 更大。

已有报告中的“约 626 倍快”是单结构、特定运行条件的比值，不适合作为普遍结论。10 结构 manifest 中，ChargE3Net 生成时间约 77–351 s，中位数 208.3 s；compact 中位数约 0.507 s，显示优势但也显示时间存在较大波动。应统一 GPU、精度、warm-up、chunk、数据读写范围，报告中位数/P90、峰值显存和不同系统大小的曲线。

**最关键的应用缺口：SCF 初猜不是只有 FFT 网格**

当前 ml、charge3net、oracle_smooth 共享已经收敛的 oracle_full PAW 单中心占据。这样的对照能隔离平滑密度质量的影响，但不是从未知结构出发的可部署初始化。[仓库协议](<C:/Users/User/Desktop/density EQGNN/reports/nmc_scf_validation_protocol.md>)已经正确说明这一点。ChargE3Net 原文也明确使用参考或另行自洽计算的 PAW augmentation；它的公开结果不能自动当作“完全无 oracle”的部署证据。[ChargE3Net Methods](https://arxiv.org/html/2312.05388v2#S4.SS4)

另外，SCF 准备脚本的 expected_electrons 来自目标 CHGCAR 网格均值。电子数通常可由结构、POTCAR ZVAL 和体系电荷获得，不需要预测，但部署入口应从这些输入获得 NELECT，而非读取答案。当前非负截断和电子数归一化只在导出阶段进行，应同时报告原始场与后处理场的误差及电子数偏差。[代码位置](<C:/Users/User/Desktop/density EQGNN/scripts/prepare_nmc_scf_benchmark.py:133>)；[VASP CHGCAR 格式及积分定义](https://vasp.at/wiki/index.php/CHGCAR)

建议把两类实验单独报告：

| 实验 | 初始输入 | 回答的问题 |
|---|---|---|
| 受控诊断 | 真值 PAW + SAD/ML/ChargE3Net/真值平滑网格 | 平滑场本身对 SCF 有多大帮助？ |
| 可部署比较 | 原生 SAD；ML 网格 + 实际可生成的 PAW；必要时共同采用同成本的短预热 | 没有最终 SCF 信息时，是否真正节省总时间？ |

若使用短 VASP 预热获得占据，必须把预热计入时间，并与同样预热的非 ML 对照比较。进一步联合预测平滑密度与 PAW 占据可以形成研究问题，但需要处理赝势版本、投影子基底、对称性和物理约束，不能假设从平滑网格唯一恢复占据。VASP 的 CHGCAR 本来就同时保存这两部分；ICHARG=1 的重启还依赖 LMAXMIX 等设置。[VASP ICHARG 文档](https://vasp.at/wiki/index.php/ICHARG)

主终点应为 T_total=T_model+T_reconstruction+T_IO+T_prewarm+T_SCF，同时报告配对电子步数、收敛失败率、最终能量一致性，以及不能忽略的最差样本。初始化密度接近真值并不意味着随机初始轨道可以零步收敛；oracle 对照未加速时应检查协议、轨道初始化和最小电子步数，不宜直接判定模型无效。

**直接相关工作，以及对本项目的影响**

| 工作与状态 | 已核实的关键内容 | 建议如何使用 |
|---|---|---|
| [SALTED，系列工作及作者代码](https://github.com/andreagrisafi/SALTED) | 原子中心径向函数与球谐展开，使用对称适配核学习密度及响应 | 原子基系数路线的基础先例；参考其 density fitting 与 overlap 处理 |
| [ChargE3Net，npj Computational Materials 2024](https://www.nature.com/articles/s41524-024-01343-1) | 高阶等变 probe 网络；NMC 文献均值约 0.060% | 保留为精度参照，明确训练集、完整网格和 PAW 协议 |
| [SCDP / A Recipe for Charge Density Prediction，NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/12c118ef87fde56a10bd858842781b34-Paper-Conference.pdf) | 虚拟中心、even-tempered Gaussian、可学习指数尺度与高阶等变网络 | 当前最直接的表示设计参照；“增加键中心、学习宽度”已有先例 |
| [ELECTRA，NeurIPS 2025；2026-01 修订版](https://arxiv.org/abs/2503.08305) | 预测浮动 Gaussian 的位置、权重和协方差，兼顾密度对称性 | 必须纳入紧凑表示的速度/精度对照；不能把浮动基单独称为新方法 |
| [BOA，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/718573eff1cb169316783d3e08514b5b-Abstract-Conference.html) | 将基函数重叠关系引入消息传递，并测试更大分子泛化 | 借鉴函数空间度量；提醒不能只把系数当作普通独立输出通道 |
| [SCFbench / Towards A Transferable Acceleration Method for DFT，2026 修订版](https://arxiv.org/abs/2509.25724) | 等变预测紧凑辅助基密度，检验尺寸迁移与 SCF 初始化 | 与项目的应用目标高度重叠；需用真实 SCF 和迁移性区分贡献 |
| [EAC-Net，JCTC 2026](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.6c00283) | 用原子贡献与实空间耦合连接基组和网格方法，覆盖多类材料 | 周期材料上的直接相关工作；适合参照轻量连续 decoder |
| [NeuralSCF，npj Computational Materials，2026-05-30](https://www.nature.com/articles/s41524-026-02110-0) | 学习密度到密度的 KS 映射，利用 SCF 轨迹训练并求固定点 | 中长期迭代修正方向；需要额外轨迹数据，当前不是最低成本入口 |
| [OrbFlow，2026-10-02 预印本](https://arxiv.org/abs/2610.02651) | 在 GTO 系数空间使用等变 flow matching | 最新跟踪项；尚不应为追新而立刻引入多步生成开销 |

上述模型的训练规模、全电子/赝密度目标、元素集合、分子/周期边界、评测网格及 SCF 后端不同，不能把各论文最优误差直接排成同一榜单。对本项目而言，最先细读并复现实验组件的顺序建议是 **SCDP → SCFbench → EAC-Net/ELECTRA → BOA**。SCFbench 的作者仓库已发布模型权重、划分和完整评估代码，可作为独立分子端验证支线。[作者发布页](https://huggingface.co/datasets/ByteDance-Seed/SCFbench)

**建议聚焦的三个方向**

**方向 A：多尺度紧凑基的周期密度表示，作为近期主线。** 保留原子图只运行一次的形式，依次检验：正确 PBC；原点正则的 l 相关径向函数；元素相关基；固定原子参考密度加残差；最后再引入少量键中心或环境相关尺度。对每种表示先做相同验证结构上的 full-grid 系数拟合，再决定是否投入 GNN 训练。

推荐形式为 ρ=ρ_ref+Σ_i,n,l,m c_i,n,l,m φ_i,n,l,m，其中所有周期像求和，尺度参数是旋转标量、同一 l 的所有 m 共用尺度。固定原子参考加残差不要求把系数解释成唯一原子电荷。新颖性应建立在周期材料上的表示效率、稳定性、跨组成泛化和实际应用证据上，而不是这些已有组件的名字。

**方向 B：针对 SCF 有用的误差设计，作为差异化重点。** 先测 full-grid NMAE、总电子数偏差、低波矢密度误差/Hartree 势误差与 SCF 步数的关系，再决定损失。可以研究 L1+电子数约束+低波矢损失，其中 G=0 单独控制、G≠0 使用有物理依据的频率权重。长程屏蔽或全局电荷模块只有在组成外推、金属性或电荷转移测试暴露问题后再加。收益目前是待验证假设，不应先宣称 NMAE 降低必然加速。

**方向 C：无真值 PAW 的实用初始化，作为更强但更难的课题。** 比较固定原子占据、统一短预热、占据预测等可获得方案，保持相同最终能量和失败率要求。如果能证明在组成外推、不同超胞和扰动结构上仍有净时间收益，这比仅在随机划分 NMC 上继续压低 NMAE 更有说服力。磁性/DFT+U 是后续扩展：当前 ISPIN=1 的基准不足以支持对一般电池材料电子态的结论。

**下一轮实验的执行顺序与判断标准**

| 阶段 | 最小工作包 | 进入下一阶段前需要看到什么 |
|---|---|---|
| 1：正确性 | 修原子图及 decoder 的周期像；补原胞/超胞与斜晶胞测试；统一推理接口 | 表示一致性达到可解释的数值容差；不再出现非物理的自身像遗漏 |
| 2：表示诊断 | 从训练/验证侧固定选 10–20 个结构，比较原基、正则基、元素基和原子参考残差；l/radial 消融逐步进行 | 同结构 full-grid 拟合误差显著下降，求解充分；若仍高，暂缓扩大 GNN |
| 3：学习诊断 | 修正后的基础模型；动态全网格采样；固定 step 调度；3 seeds；同预算逐项消融 | 量化模型与逐结构拟合的差距，分清表示和学习瓶颈 |
| 4：受控 SCF | 已准备 10 结构作协议检查，再固定更大代表性集合 | 初始化确实被读取；失败率、最终能量及总时间可比较 |
| 5：部署与外推 | 去掉 oracle PAW；加入组成/Li 区间/结构簇与超胞测试 | 净时间收益及配对统计有支持，且没有明显可靠性退化 |

阶段 2 可以把“将约 0.67% 的数值拟合误差降到约 0.2% 或更低”作为内部探索目标，但它不是预测、保证或论文标准，也不能替代阶段 4–5。若更高表示精度不能改善真实 SCF，就应转向初始化敏感的误差方向，而不是继续堆 l_max。

近期不建议优先做大规模 MP 重训练、单纯增深 backbone、直接照搬 flow matching，或再次以病态 LS 系数 MSE 作为唯一目标。现阶段最有信息量的投入是 **修正确性 → 对齐表示诊断 → 完成可部署 SCF 闭环**。本次未改动模型和训练源码；改进收益仍需上述实验确认。
