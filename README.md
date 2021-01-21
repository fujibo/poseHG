pose-hg-demo (reimplementation)
=====

I reimplemented "Stacked hourglass networks for human pose estimation" [1] mainly following the author code.
This paper provides an **only single-person pose estimation**. If you want to do a multi-person pose estimation like Openpose, you need to combine this with a person detector like SSD [2]. We provide the code, `demo_multi_person.py`.

## Requirements
- python==3.6.4
- chainer==4.0.0
- chainercv==0.9.0
- opencv-python==3.4.0
- wget
- cupy==4.0.0 (if you want to use your gpu)

## NOTE
**This code is not perfectly same with the original code.**
The known differences are as follows:
- We used resizing (bilinear) instead of nearest neighbor upsampling for upsampling. This is because chainer didn't provide NNupsampling.
- We used the same images but **didn't use the same validation data they provide.** They use 1-4 people in each image while we use only a person for evaluation. **Therefore, we reevaluated scores on our validation set.**
- The network architecture they provided on github might not be identical with the one of a pre-trained model (a t7 file) they provide in their project page. For example, conv-bn-relu is used in their github code, while relu-bn-conv seems to be used in their t7 file.

## How to execute
A pre-trained model is utilized in the default settings. If you want to specify a model, you need to add `--model <model_path>` as an argument.
We only provide chainer version.

```bash
cd chainer
```

### Demo
#### single person
If you set filename like sample.jpg, you can get input.jpg, output.jpg. *If you want to run in gpu mode, you just need to set `--gpu <GPU ID>`*.

expected properties
- A person is in the center of the image
- the height of this image == 1.25 * a person's scale (= height)

```bash
python demo.py --image filename
```

### multi person
If you have some people in an image, you have an option to detect people followed by estimating poses.

```bash
python demo_multi_person.py --image filename
```

we utilized SSD512 [2] implemented in chainercv for detecting people.

### evalution
```bash
python eval.py
```
The metric is PCKh@0.5.

### Training
You need to download images of MPII at `./chainer/data/MPII/images` and annotations of it at `./chainer/data/MPII/annotations`.
Training will take 2-4 days.
```bash
python train.py --gpu 0 --out results/result
```
You can get a trained model at `./results/result/model.npz`.

## Results
Comparison between the neural network weights of theirs and those of ours on our validation data. `+flip` denotes flipping as a test time augmentation.

|         | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Total |
| :---:   |:---: | :---:    | :---: | :---: |:---:|:---: | :---: | :---: |
|our val, their weight | 0.975 | 0.960 | 0.901 | 0.861 | - | 0.853 | 0.823 | 0.896 |
|our val, our weight |**0.952**|**0.930**|**0.866**|**0.821**|**0.839**|**0.814**|**0.790**| - |
|our val, our weight + flip | 0.955 | 0.935 | 0.874 | 0.828 | 0.854 | 0.827 | 0.801 | - |

<!-- |original test | 0.982 | 0.963 | 0.912 | 0.871 | 0.901 | 0.876 | 0.836 | - | -->
<!-- |their val/their weight | 0.968 | 0.952 | 0.891 | 0.842 | - | 0.832 | 0.804 | 0.881 | -->


## Road for Reproduction
- [x] metrics
- [ ] check pre-process (by saving annot file by myself)
- [ ] check weight (train)
- [ ] pytorch version

## Reference
- [1] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In ECCV, pp. 483–499. Springer, 2016.
- [2] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. Ssd: Single shot multibox detector. In ECCV, pp. 21–37, 2016.
