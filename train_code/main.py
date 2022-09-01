# coding=gbk

import argparse
import xml.etree.ElementTree as ET
from utilis import *
import argparse

label_list = ['head']

def get_image_txt(opt):

    #���׶�һ���������ݼ�������ϴ����
    # ��һ��������images_label_split�е�ͼ��ɾ�������xml
    print("V1")
    compare_image_label_remove_xml(opt.train_data)
    # # # �ڶ���������images_label_split�е�ͼ��ɾ�������image
    # print("V2")
    compare_image_label_remove_image(opt.train_data)
    # # ���������������ļ����е�xml�������������ļ�ɾ��
    print("V3")
    remove_not_satisfied_xml(opt.train_data)
    # # ���Ĳ�������xml�Ƿ�Ϊ�գ��յĻ�ɾ��xml,Ҳɾ����Ӧ��image
    print("V4")
    remove_image_null_xml(opt.train_data,label_list)
    # # ���岽������image��xml�����ݣ���ʾͼƬ�����ÿ��Ƿ���ȷ
    # show_label(opt.train_data,label_list)

    #���׶ζ��������ݰ���һ�������ֳ�ѵ������֤����
    # ��train��test����ֿ�����image��xml�ֱ𱣴浽train��test���ڵ��ļ�����
    # ����ǰ����Եõ�xml��image,ÿ��������ѡ��10%������,��Ϊ��֤��, ����train��test�����ļ���
    yolov3_get_train_test_file(opt.train_data,0.2)

    # �׶�������train��test��xml��ת����txt
    # ��һ������train��test�е�xml�ļ�����txt�ļ������ŵ�image_txt�ļ�����
    yolov3_get_txt(opt.train_data,label_list)
    # #  �ڶ����������е�image�ļ�һ���ƶ���image_txt��
    yolov3_move_image(opt.train_data)
    # # ����������train/Annotations��test/Annotations��xml�Զ�����train.txt��test.txt�ļ��������浽train_test_txt��
    yolov3_get_train_test_txt(opt.train_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='..\\data\\head_train_data', help='data dir')
    opt = parser.parse_args()
    get_image_txt(opt)
