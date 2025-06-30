## **Re-Diffusion：Modeling Latent Residuals with Diffusion for Time-Series Forecasting
**

### **About Re-Diffusion 💡**

Re-Diffusion serves as a time-series forecasting tool to aid in promoting long-term sequence estimation. It constructs on residuals that rely on different backbones, which makes it conventional as a plug-in-and-out tool, including backbones:

1. point estimation network like iTransformer, AutoFormer etc.
2. probalistic estimation network like CSDI etc.
3. Othe long-term forecasting methods.

Residuals are estimated by a latent diffusion model which includes several potential conditions. As defined in the work, timestamps and history knowledge are both useful in the design of a condition-aware network. 

### **Usage 🔧**
Re-Diffusion is developed with Python 3.10.Install Pytorch and the necessary dependencies.
```python
pip install requirement.txt
```

The datasets can be obtained from [GoogleDrive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view)

We include backbones：

| model | estimation |
| ------- | ------- |
|    [Autoformer](https://arxiv.org/abs/2106.13008)     |    point     |
|	[Informer](https://arxiv.org/abs/2012.07436)	| point |
|	[iTransformer](https://arxiv.org/abs/2310.06625)| point	|
|	[PatchTST](https://arxiv.org/abs/2211.14730)	|	point	|
|	[TimeMixer](https://arxiv.org/abs/2405.14616)	|	point	|
|	[TimeXer](https://arxiv.org/abs/2402.19072)	|	point	|
|	[CSDI](https://arxiv.org/abs/2107.03502)	|probablistic	|
|	[TimeGrad](https://arxiv.org/abs/2101.12072)	|	probablistic	|
|	[TSDiff](https://arxiv.org/abs/2307.11494)	| probablistic	|

First, we need to train the backbone-models to predict normal distribution(autoformer as an example)
```python
cd multi-forecasting
bash autoformer.sh
```
Then, we train the corresponding residual vae model:
```
python vae.py --model autoformer
```

At last, we include diffusion model to predict:
```
python re-diffusion.py --model autoformer
```




