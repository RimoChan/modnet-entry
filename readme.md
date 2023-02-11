# 【MODNet-entry】开箱即用的人像抠图工具


前几天我用stable-diffusion生成了几千张萝莉图片，准备用来做游戏立绘，但是它出的图都是带背景的……也不是不能抠啦，但是我一点手工活都不想做，所以就在GitHub上找了找有没有什么全自动抠人的模型。

但是我找到的都没有方便的接口，一般都是给一个模型，然后给一堆代码，要自己加载自己调，很麻烦。所以我就给MODNet包了一层，可以直接用pip安装，这下就方便了！

模型效果可以看[原仓库](https://github.com/ZHKKKe/MODNet)，我就不复制图片过来啦。


## 安装

```bash
pip install git+https://github.com/RimoChan/modnet-entry.git
```

安装时会从Google Drive下载预训练模型，所以要保证你的网络是好的。


## 接口

```python
def get_model(ckpt_name: str) -> MODNet: ...
```

获取一个预训练的模型。

参数: 

- `ckpt_name`: 模型的名字。只有`modnet_photographic_portrait_matting.ckpt`/`modnet_webcam_portrait_matting.ckpt`两种可选。


```python
def infer(modnet: MODNet, im: np.ndarray[np.uint8], ref_size=1024) -> np.ndarray[np.float32]: ...
```

输入一张图，预测alpha通道。

参数: 

- `modnet`: 刚才加载的那个模型。

- `im`: 图片。RGB或RGBA或灰度的uint8矩阵。

- `ref_size`: 预测时如果图片的短边长于这个尺寸就缩小到这个尺寸。

返回一个与原图相同大小的灰度float矩阵。


```python
def infer2(modnet: MODNet, img_path: str, out_alpha_path: str = '', out_img_path: str = ''): ...
```

输入一个图片路径，将抠图结果保存在硬盘上。

参数: 

- `modnet`: 刚才加载的那个模型。

- `img_path`: 输入图片路径。

- `out_alpha_path`: 输出alpha图片路径。

- `out_img_path`: 输出抠好的图的路径。


## 结束

就这样，我要去看萝莉图片了，大家88！
