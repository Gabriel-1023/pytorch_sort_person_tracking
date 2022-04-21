
实现了较高帧率下的行人跟踪功能

行人识别来自：[ziweizhan/person-detector](https://github.com/ziweizhan/person-detector)  
SORT跟踪来自：[abewley/SORT](https://github.com/abewley/sort)  
 
具体安装和模型训练方法可参考[ziweizhan/person-detector](https://github.com/ziweizhan/person-detector) 中的README.md  


Exception: initMNN: init numpy failed  
```bash
pip uninstall numpy
pip install numpy
```

MNN.Tensor() 卡住 exit code -1073740791 (0xC0000409)  
MNN.Tensor()前加入  
```python
image = image.astype(np.float32)
```

