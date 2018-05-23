pose-hg-demo (reimplimentation)
=====

I reimplimented "Stacked" mainly following this author code.
This paper provides **only single-person pose estimation**. If you want to do multi-person pose estimation like Openpose, please combine with detector like SSD.

## Requirements
- python==3.6.4
- chainer==4.0.0
- chainercv==0.9.0
- opencv-python==3.4.0

## NOTE
**this code is not perfectly same with the original code.**
the known differences are as follows:
- we used resizing (bilinear) instead of nearest neighbor upsampling. this is because chainer didn't provide NNupsampling.
- we used the same images but didn't use the same validation data they provide. They use 1-4 people in each image while we use only a person for evaluation. **This may cause a little difference on scores**.

Nevertheless, we scores almost same scores with them on MPII.
The metric is PCKh@0.5

## How to execute
pre-trained model is available at here. This will take several minutes.

### evalution
```bash
python eval.py --gpu 0 --model ./model.npz
```

### demo
if you set filename like sample.jpg, you can get input.jpg, output.jpg. *If you want to run in cpu mode, you just set `--gpu -1`*.

expected properties
- A person is in the center of the image
- the height of this image == 1.25 * a person's scale (= height)

```bash
python demo.py --gpu 0 --image filename --model ./model.npz
```

Training will take 2-4 days.
```bash
python train.py --gpu 0 --out results/result
```

## Results
here comes a table soon.

## Future work
- [x] combine with detection (as an application).

## Road for Reproduction
- [x] metrics
- [ ] check pre-process (by saving annot file by myself)
- [ ] check weight (train)
