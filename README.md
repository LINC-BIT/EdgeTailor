# EdgeTailor

本项目实现了一个面向边缘侧动态场景的长尾持续学习与域自适应方法 **EdgeTailor**。  
该方法旨在解决实际部署中存在的类别不均衡（long-tailed distribution）、任务连续到达（continual learning）以及跨域分布偏移（domain shift）等问题。

整体框架包括两个核心部分：

- **EdgeTailorCL**：用于提升长尾持续学习性能，缓解灾难性遗忘问题  
- **EdgeTailorDA**：用于提升模型在跨域场景下的泛化能力  

---

## 🚀 1. 环境配置

```bash
conda create -n edgetailor python=3.8
conda activate edgetailor

pip install -r requirements.txt
