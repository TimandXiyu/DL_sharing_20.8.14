import os
import sys
import numpy as np
import cv2 as cv
import torch
file_path=__file__
ALL_PATH=os.path.dirname(file_path)+"/../"

from Control_BigHomework.Network.Message import MessageProcesser

ALL_PATH=os.path.abspath(ALL_PATH)
LIB_PATH=ALL_PATH+"/Network/lib"
print(ALL_PATH)
sys.path.insert(0,ALL_PATH)#导入CenterNet的lib包

from Camera.Camera import Camera
import serial

LIB_PATH=os.path.abspath(LIB_PATH)
print("lib",LIB_PATH)
sys.path.insert(0,LIB_PATH)#导入CenterNet的lib包

from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH=os.path.dirname(file_path)+"/models/multi_pose_dla_3x.pth"
TASK='multi_pose'

#导入相机内参
with np.load("./camera_matrix.npz") as data:
    camera_matrix=data['camera_matrix']
    fx=camera_matrix[0,0]
    cx=camera_matrix[0,2]
    fy=camera_matrix[1,1]
    cy=camera_matrix[1,2]


class PoseDetector:
    def __init__(self):
        #生成识别的opt
        self.opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
        self.detector = detector_factory[self.opt.task](self.opt)
        self.detections=None
        #绘制关节点
        self.colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                          (255, 0, 0), (0, 0, 255)]
        #绘制边缘线
        self.ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
                   (255, 0, 0), (0, 0, 255), (255, 0, 255),
                   (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]

        #对应要连线的边缘
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [3, 5], [4, 6], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [5, 11], [6, 12], [11, 12],
                      [11, 13], [13, 15], [12, 14], [14, 16]]

    def detect(self,image):
        """
        用于进行目标识别
        """
        self.detections=self.detector.run(image)['results']
        return self.detections

    def filter_result(self,detections=None,confidence=0.4):
        """
        用于过滤掉识别较差的结果
        """
        if detections is None:
            detections=self.detections

        filtered_results=[]
        for result in detections[1]:
            if result[4]>confidence:
                filtered_results.append(result)

        return filtered_results

    def show_results(self,results,image):
        for result in results:
            x1,y1,x2,y2=result[:4]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            state=self.extract_motion(result)
            if state==1:
                cv.putText(image,"lay",(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            elif state==2:
                cv.putText(image,"stand",(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            elif state==3:
                cv.putText(image,"sit",(x1,y1),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            points=result[5:]
            points=np.array(points,dtype=np.int32).reshape(17,2)#一共有17个点
            for j in range(17):
                #画点
                cv.circle(image,(points[j, 0], points[j, 1]), 5, self.colors_hp[j], -1)
                cv.putText(image,str(j),(points[j, 0], points[j, 1]),cv.FONT_HERSHEY_SIMPLEX,0.8,self.colors_hp[j])

            for j,e in enumerate(self.edges):
                cv.line(image, (points[e[0], 0], points[e[0], 1]),(points[e[1], 0], points[e[1], 1]), self.ec[j], 2)

    @staticmethod
    def is_verticle(point1,point2,point3=None,point4=None):
        """
        送入两个点,判断这两个点是横着的还是竖着的
        如果竖着的,返回True,横着的,返回False
        """
        x1,y1=point1
        x2,y2=point2

        x_dis=abs(x1-x2)
        y_dis=abs(y1-y2)

        if (point3 is not None) and (point4 is not None):#四个点就同时判断
            x3,y3=point3
            x4,y4=point4
            x_dis1=abs(x3-x4)
            y_dis1=abs(y3-y4)
            x_all=x_dis+x_dis1
            y_all=y_dis+y_dis1
            if x_all<y_all:
                return True
            else:
                return False

        if x_dis<y_dis:#y方向的差距更大,横着的
            return True
        else:#x方向差距大,是竖着的
            return False

    def extract_motion(self,result):
        """
        送入识别的结果,判断这个人的对应姿态
        通过获取人体的姿态,确定人是坐着的还是躺着的还是站立的
        如果肩膀和大腿根部是竖着的,则是坐着或者站立
        如果膝关节和大腿根部是平着的,则是坐着
        1:是平躺着的
        2:是站着的
        3:是躺着的
        """
        points=result[5:]
        points=np.array(points,dtype=np.int32).reshape(17,2)#或许人体的17个点,从0开始的

        #获取多个关节点
        Lsho=points[5]#获取肩膀点
        Rsho=points[6]


        Relb=points[8]#获取手肘点
        Lelb=points[7]

        Lwri=points[9]#获取手掌点
        Rwri=points[10]

        Lhip=points[11]#获取大腿根部
        Rhip=points[12]

        Lkne=points[13]#获取膝关节
        Rkne=points[14]

        lay_flag=self.is_verticle(Lsho,Lhip,Rsho,Rhip)
        if lay_flag:

            sit_flag=self.is_verticle(Lhip,Lkne,Rhip,Rkne)
            if sit_flag:
                return 2
            else:
                return 3

        else:
            return 1

    def detect_with_motion(self,image,depth_image,show_result=True):
        """
        送入一张图片,最终返回一个list,每个list包含人的xyz和对应姿态的分类
        """
        #1:进行目标识别
        detections=self.detector.run(image)['results']
        filtered_results=self.filter_result(detections,confidence=0.8)
        if show_result:
            self.show_results(filtered_results,image)

        with_pose_results=[]

        #2:处理每个results
        for result in filtered_results:
            x1,y1,x2,y2=result[:4]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

            #2.1:解析动作
            state=self.extract_motion(result)

            #2.2:获取xyz
            x,y,z=self.get_XYZ(depth_image,point=(int((x1+x2)/2),int((y1+y2)/2)))

            with_pose_results.append((x,y,z,state,(int((x1+x2)/2),int((y1+y2)/2))))

        return with_pose_results

    def get_point(self,image,point):
        """
        用于索引图片点,避免numpy和opencv的点的位置混乱
        :param image:
        :param point:
        :return:
        """
        if len(image.shape)==2:
            return image[point[1],point[0]]
        if len(image.shape)==3:
            return image[point[1],point[0],:]

    def get_Z_from_range(self,depth_image,point,rect_w=70,rect_h=50):
        """
        为了避免一个区域没有值,因此直接取这个区域的中值作为数据获取
        """
        point_x,point_y=point
        roi=depth_image[point_y-rect_h:point_y+rect_h,point_x-rect_w:point_x+rect_w]
        roi=roi.reshape(-1,1)
        roi=roi[roi!=0]
        Z=np.mean(roi)
        return Z

    def get_XYZ(self,depth_image,point):
        Z=self.get_point(depth_image,point)
        if Z==0:
            #出现搜索框的中心点没有人的情况
            Z=self.get_Z_from_range(depth_image,point)
            if Z==np.nan:
                Z=0


        X=(point[0]-cx)*Z/fx
        Y=(point[1]-cy)*Z/fy
        return X,Y,Z

    def show_motion_result(self,motion_results,color_image):
        for result in motion_results:
            x,y,z,state,center=result
            cv.circle(color_image,center,3,(0,0,255),2)
            cv.putText(color_image,"{:.0f},{:.0f},{:.0f},{}".format(x,y,z,state),center,cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)


if __name__ == '__main__':
    #初始化摄像头类
    camera=Camera(2)#如果是台式机,默认为0,如果是笔记本,打开笔记本自带摄像头的时候,就0和2之间切换

    #设置姿态识别
    pose_detector=PoseDetector()

    bps=115200
    USB0=serial.Serial(port='/dev/ttyUSB0',baudrate=bps,timeout=0.1)

    messageProcesser=MessageProcesser()
    x=0

    while True:
        place1=[]
        place2=[]
        place3=[]
        print("*******************开始进行检测************************************")
        #1:读取图片
        depth_image,color_image=camera.get_pictures()

        #2:获取姿态
        """
        motion_results是一个list,为:[[x,y,z,state,center],[x,y,z,state,center],[...],...],每一个代表一个人
        x,y,z是相机坐标系下的目标位置,z可能出现是nan的情况,这时候尽量离摄2m远即可
        state中1:是平躺着的,2:是站着的,3:是坐着的
        center是图片中的点
        """
        motion_results=pose_detector.detect_with_motion(color_image,depth_image,show_result=True)#获取到每个结果


        #2.1:解析姿态信息:
        for result in motion_results:
            x,y,z,state,center=result#获取x,y,z,state,center
            if z>5000:
                place1.append(state)
            if 2000<z<5000:
                place2.append(state)
            if z<2000:
                place3.append(state)

            print("某人的位置为:",x,y,z,state)


        #2.2:基于区域生成发送数据
        big_state_1=4
        for people_state in place1:
            if people_state<big_state_1:
                big_state_1=people_state

        big_state_2=4
        for people_state in place2:
            if people_state<big_state_2:
                big_state_2=people_state


        big_state_3=4
        for people_state in place3:
            if people_state<big_state_3:
                big_state_3=people_state



        #2.3:生成发送数据
        # send_message=messageProcesser.get_send_msg(x1=len(place1),x3=big_state_1,x5=len(place2),x6=big_state_2,x7=len(place3),x9=big_state_3,x10=13,x11=10)
        send_message=messageProcesser.get_send_msg(x2=len(place1),x3=big_state_1,x4=len(place2),x5=big_state_2,x6=len(place3),x7=big_state_3,x10=13,x11=10)
        print(len(place1),big_state_1,len(place2),big_state_2,len(place3),big_state_3)


        # send_message="{},{}/{},{}/{},{}".format(len(place1),big_state_1,len(place2),big_state_2,len(place3),big_state_3)
        # print("发送数据为:",send_message)
        USB0.write(send_message)
        cv.namedWindow("color_image",cv.WINDOW_NORMAL)
        cv.imshow("color_image",color_image)
        cv.waitKey(1)


