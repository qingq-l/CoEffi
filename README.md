# CoEffi

![CoEffi](Figures/CoEffi.png)

Welcome to CoEffi, a dynamic collaborative decoding framework enhancing the execution efficiency of LLM-generated code.

---

## 📁 Project Structure

```
CoEffi/
├── Data/                          # Training dataset
├── Figures/                       # Figures
├── MainEvalResults/               # Evaluation results
├── Mercury-main/                  # Mercury benchmark
├── enamel/                        # ENAMEL benchmark
├── CustomizedGeneration.py        # CoEffi implementation
├── train_SkelDPO.py               # SkelDPO training script
├── run_eval_mercury.py            # Standard evaluation on Mercury
├── run_eval_enamel.py             # Standard evaluation on ENAMEL
├── run_eval_effibench.py          # Standard evaluation on EffiBench
├── run_collab_eval_mercury.py     # CoEffi evaluation on Mercury
├── run_collab_eval_enamel.py      # CoEffi evaluation on ENAMEL
├── run_collab_eval_effibench.py   # CoEffi evaluation on EffiBench
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

For EffiBench, please download it separately from:
`https://github.com/huangd1999/EffiBench`

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. SkelDPO Training

```bash
python train_SkelDPO.py \
    --model_path <path_to_base_model> \
    --train_path Data/TrainingDatasetForSkeldpo.json \
    --save_dir <output_dir>
```

(Additional training arguments can be found in the script)

### 2. Standard Evaluation

#### On Mercury:
```bash
python run_eval_mercury.py
```

#### On ENAMEL:
```bash
python run_eval_enamel.py
```

#### On EffiBench:
```bash
python run_eval_effibench.py
```

### 3. CoEffi Evaluation

#### On Mercury:
```bash
python run_collab_eval_mercury.py
```

#### On ENAMEL:
```bash
python run_collab_eval_enamel.py
```

#### On EffiBench:
```bash
python run_collab_eval_effibench.py
```

---

## 🙏 Acknowledgements & References

This project builds upon the following excellent works:

- **Mercury**: https://github.com/Elfsong/Mercury
- **ENAMEL**: https://github.com/q-rz/enamel
- **EffiBench**: https://github.com/huangd1999/EffiBench
- **SkelDPO**: https://github.com/YYYY-YuYu/SkelDPO
