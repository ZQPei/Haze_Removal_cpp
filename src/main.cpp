#include "hazeremoval.h"
#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **args) {
	const char * img_path = args[1];
	Mat in_img = imread(img_path);
	Mat out_img(in_img.rows, in_img.cols, CV_8UC3);
	unsigned char * indata = in_img.data;
	unsigned char * outdata = out_img.data;

	CHazeRemoval hr;
	cout << hr.InitProc(in_img.cols, in_img.rows, in_img.channels()) << endl;
	cout << hr.Process(indata, outdata, in_img.cols, in_img.rows, in_img.channels()) << endl;
	imshow("out_img", out_img);
	waitKey(0);
	// system("pause");
	return 0;
}