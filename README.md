# Haze Removal

## Introduction

This is an simple implemention of "single image haze removal using dark channel prior" by kaiming He, who wins the CVPR09 best paper.

## Parameters by default
- radius=7
- omega=0.95
- t0=0.1
- r=60
- eps=0.001

## Dependencies

- opencv>=2.4
- g++ (Visual Studio is also fine, but I will not show how to configure VS project here)

## Compile

```bash
cd cpp_code/src
g++ main.cpp hazeremoval.cpp guidedfilter.cpp -o ../dehaze `pkg-config --libs --cflags opencv`
```

## Run

Before running, you should check whether your LD_LIBRARY_PATH containing your opencv lib path!

```bash
cd ..
./dehaze [your image path]
```


## DEMO

<figure class="half">
    <img src="demo/canon3.bmp">
    <img src="demo/canon3_rev.jpg">
</figure>

<figure class="half">
    <img src="demo/22.jpg">
    <img src="demo/22_rev.jpg">
</figure>


## References

- paper: Single Image Haze Removal using Dark Channel Prior
- paper: Guided Image Fltering
