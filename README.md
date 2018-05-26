pose-hg-demo (reimplementation)
=====

I reimplemented "Stacked hourglass networks for human pose estimation" [1] mainly following this author code.
This paper provides **only single-person pose estimation**. If you want to do multi-person pose estimation like Openpose, you need to combine with detector like SSD [2]. We provide the code.

## Requirements
- python==3.6.4
- chainer==4.0.0
- chainercv==0.9.0
- opencv-python==3.4.0

## NOTE
**this code is not perfectly same with the original code.**
the known differences are as follows:
- we used resizing (bilinear) instead of nearest neighbor upsampling. this is because chainer didn't provide NNupsampling.
- we used the same images but **didn't use the same validation data they provide.** They use 1-4 people in each image while we use only a person for evaluation. **Therefore, **we reevaluated scores on my validation set.**

Nevertheless, we scores almost same scores with them on MPII.
The metric is PCKh@0.5

## How to execute
pre-trained model is available at here. This will take several minutes.

### Demo
if you set filename like sample.jpg, you can get input.jpg, output.jpg. *If you want to run in cpu mode, you just set `--gpu -1`*.

expected properties
- A person is in the center of the image
- the height of this image == 1.25 * a person's scale (= height)

```bash
python demo.py --gpu 0 --image filename --model ./model.npz
```

Or, if you have some people in an image, you have an option to detect people followed by estimating poses.

```bash
python demo_multi_person.py --gpu 0 --image filename --model ./model.npz
```

we utilized SSD512 [2] for detecting people.

### Training
Training will take 2-4 days.
```bash
python train.py --gpu 0 --out results/result
```

### Evalution
```bash
python eval.py --gpu 0 --model ./model.npz
```

## Results
Comparison between the neural network weights of theirs and those of ours on my validation data. `+flip` denotes flipping as test time augmentation.

|         | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Total |
| :---:   |:---: | :---:    | :---: | :---: |:---:|:---: | :---: | :---: |
|my val, their weight | 0.975 | 0.960 | 0.901 | 0.861 | - | 0.853 | 0.823 | 0.896 |
|my val, my weight |**0.952**|**0.930**|**0.866**|**0.821**|**0.839**|**0.814**|**0.790**| - |
|my val, my weight + flip | 0.955 | 0.935 | 0.874 | 0.828 | 0.854 | 0.827 | 0.801 | - |

<!-- |original test | 0.982 | 0.963 | 0.912 | 0.871 | 0.901 | 0.876 | 0.836 | - | -->
<!-- |their val/their weight | 0.968 | 0.952 | 0.891 | 0.842 | - | 0.832 | 0.804 | 0.881 | -->


## Road for Reproduction
- [x] metrics
- [ ] check pre-process (by saving annot file by myself)
- [ ] check weight (train)

## Reference
- [1] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In ECCV, pp. 483–499. Springer, 2016.
- [2] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In ECCV, pp. 21–37, 2016.
