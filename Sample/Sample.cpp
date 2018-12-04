#include "ZQ_FastFaceDetector.h"
#include <vector>
#include <iostream>
#include <omp.h>
#include "opencv2/opencv.hpp"
#pragma comment(lib,"mklml.lib")
#pragma comment(lib,"ZQCNN.lib")
#pragma comment(lib,"ZQ_FastFaceDetector.lib")
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
			circle(image, Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 3, Scalar(0, 255, 255), -1);
	}
}

int main()
{
	vector<face_box> result;
	string result_name;
	Mat image0 = imread("data\\4.jpg", 1);
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}

	result_name = "result.jpg";
	int thread_num = 4;
	if (!init(thread_num, "test2000times"))
	{
		cout << "failed to init\n";
		return EXIT_FAILURE;
	}

	int iters = 100; //you can only call 2000 times
	int min_face_size = 20;
	int do_landmark = true;
	int refine_landmark = false;
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (!detect(result, image0.data, image0.cols, image0.rows, image0.step[0], min_face_size, do_landmark, refine_landmark))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
			continue;
		}
	}
	double t2 = omp_get_wtime();
	printf("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);

	namedWindow("result");
	Draw(image0, result);
	imwrite(result_name, image0);
	imshow("result", image0);

	waitKey(0);
	return EXIT_SUCCESS;
}