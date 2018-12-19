#ifndef _ZQ_FAST_FACE_DETECTOR_V3_H_
#define _ZQ_FAST_FACE_DETECTOR_V3_H_
#include <vector>

struct face_box
{
	float score;
	int row1;
	int col1;
	int row2;
	int col2;
	float ppoint[10];
};

bool init_v3(void** ptr, int thread_num, bool do_landmark, bool refine_landmark = false, const char* username = "test2000times");

bool detect_v3(void* ptr, std::vector<face_box>& result, 
	const unsigned char* bgr_img, int width, int height, int width_step, 
	int min_face_size = 60, float prob_thresh1 = 0.5, float prob_thresh2 = 0.6, float prob_thresh3=0.8, 
	int support_num = 3, bool handle_very_large_face = false);

void release_v3(void** ptr);

#endif
