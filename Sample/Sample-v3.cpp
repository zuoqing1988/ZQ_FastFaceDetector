#include "ZQ_FastFaceDetector-v3.h"
#include <vector>
#include <iostream>
#include <omp.h>
#include "opencv2/opencv.hpp"
#pragma comment(lib,"mklml.lib")
#pragma comment(lib,"ZQCNN.lib")
#pragma comment(lib,"ZQ_FastFaceDetector-v3.lib")
using namespace std;
using namespace cv;

static void Draw(Mat &image, const vector<face_box>& result)
{
	vector<face_box>::const_iterator it = result.begin();
	for (; it != result.end(); it++)
	{
		if (it->score > 0.7)
		{
			rectangle(image, Point((*it).col1, (*it).row1), Point((*it).col2, (*it).row2), Scalar(0, 0, 255), 2, 8, 0);
		}
		else
		{
			rectangle(image, Point((*it).col1, (*it).row1), Point((*it).col2, (*it).row2), Scalar(0, 255, 0), 2, 8, 0);
		}

		for (int num = 0; num < 5; num++)
			circle(image, Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 1, Scalar(0, 255, 255), -1);
	}
}

int main()
{
	vector<face_box> result;
	bool run_hard = false;
	Mat image0;
	Mat draw0, draw1, draw2;
	int min_face_size = 48;
	bool do_landmark = false;
	float thresh1 = 0.6;
	float thresh2 = 0.7;
	float thresh3 = 0.8;
	int support_num = 3;
	
	image0 = imread("data\\4.jpg", 1);
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	image0.copyTo(draw0);
	image0.copyTo(draw1);
	image0.copyTo(draw2);
	int thread_num = 1;
	void* worker0 = 0, *worker1 = 0, *worker2 = 0;
	if (!init_v3(&worker0, thread_num, false, false, "test2000times"))
	{
		cout << "failed to init\n";
		return EXIT_FAILURE;
	}
	if (!init_v3(&worker1, thread_num, true, false, "test2000times"))
	{
		cout << "failed to init\n";
		release_v3(&worker0);
		return EXIT_FAILURE;
	}
	if (!init_v3(&worker2, thread_num, true, true, "test2000times"))
	{
		cout << "failed to init\n";
		release_v3(&worker0);
		release_v3(&worker1);
		return EXIT_FAILURE;
	}

	int iters = 1000; //you can only call 2000 times
	
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (!detect_v3(worker0, result, image0.data, image0.cols, image0.rows, image0.step[0], min_face_size, 
			thresh1, thresh2, thresh3, support_num, false))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
			continue;
		}
	}
	double t2 = omp_get_wtime();
	printf("no landmark: total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);
	Draw(draw0, result);

	iters = 500;
	do_landmark = true;
	double t3 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (!detect_v3(worker1, result, image0.data, image0.cols, image0.rows, image0.step[0], min_face_size,
			thresh1, thresh2, thresh3, support_num, false))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
			continue;
		}
	}
	double t4 = omp_get_wtime();
	printf("with landmark: total %.3f s / %d = %.3f ms\n", t4 - t3, iters, 1000 * (t4 - t3) / iters);
	Draw(draw1, result);

	double t5 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (!detect_v3(worker2, result, image0.data, image0.cols, image0.rows, image0.step[0], min_face_size,
			thresh1, thresh2, thresh3, support_num, false))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
			continue;
		}
	}
	double t6 = omp_get_wtime();
	printf("refine landmark: total %.3f s / %d = %.3f ms\n", t4 - t3, iters, 1000 * (t4 - t3) / iters);
	Draw(draw2, result);

	
	namedWindow("no landmark");
	namedWindow("with landmark");
	namedWindow("refine landmark");
	
	imwrite("result0.jpg", draw0);
	imwrite("result1.jpg", draw1);
	imwrite("result2.jpg", draw2);
	imshow("no landmark", draw0);
	imshow("with landmark", draw1);
	imshow("refine landmark", draw2);

	waitKey(0);
	release_v3(&worker0);
	release_v3(&worker1);
	release_v3(&worker2);
	return EXIT_SUCCESS;
}