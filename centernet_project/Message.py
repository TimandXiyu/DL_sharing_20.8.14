"""
对位的信息传输类
"""
import serial
import serial.tools.list_ports
import struct
import time
from ctypes import c_ubyte,c_float,Structure

import threading


global DEBUG_FLAG
global SHOW_IMAGE_FLAG
global USE_SERIAL_FLAG

DEBUG_FLAG=True
SHOW_IMAGE_FLAG=False
USE_SERIAL_FLAG=False


#用于定义数据发送类型的结构体
class MessageType(Structure):
    _fields_=[
        ("x1", c_ubyte),
        ("x2", c_ubyte),
        ("x3", c_ubyte),
        ("x4", c_ubyte),
        ("x5", c_ubyte),
        ("x6", c_ubyte),
        ("x7", c_ubyte),
        ("x8", c_ubyte),
        ("x9", c_ubyte),
        ("x10", c_ubyte),
        ("x11", c_ubyte)

    ]


#定义数据发送接收类
class MessageProcesser:
    def __init__(self):
        #1:初始化串口
        bps=115200
        port_list=list(serial.tools.list_ports.comports())
        port_list.sort(key=lambda x:x.device)#确保排列顺序为USB0,USB1

        #如果一个串口,则是使用模式
        if len(port_list)==1:
            self.USB0=serial.Serial(port=port_list[0].device,baudrate=bps,timeout=0.0001)#采用0.1ms的等待间隔

        #如果两个串口,则是测试串口模式
        elif len(port_list)==2:
            self.USB0=serial.Serial(port=port_list[0].device,baudrate=bps,timeout=0.0001)
            self.USB1=serial.Serial(port=port_list[1].device,baudrate=bps,timeout=0.0001)


        #2:初始化发送通信数据
        self.messageType=MessageType()
        self.messageType.x1=c_ubyte(1)
        self.messageType.x2=c_ubyte(1)
        self.messageType.x3=c_ubyte(1)
        self.messageType.x4=c_ubyte(1)
        self.messageType.x5=c_ubyte(1)
        self.messageType.x6=c_ubyte(1)
        self.messageType.x7=c_ubyte(1)
        self.messageType.x8=c_ubyte(1)
        self.messageType.x9=c_ubyte(1)
        self.messageType.x10=c_ubyte(1)
        self.messageType.x11=c_ubyte(1)


        self.send_list=[self.messageType.x1,self.messageType.x2,self.messageType.x3,self.messageType.x4,self.messageType.x5,self.messageType.x6,self.messageType.x7,self.messageType.x8,self.messageType.x9,self.messageType.x10]
        self.send_type='B'*11
        self.send_msg=None

        #3:初始化信息接收
        self.read_msg_array=None
        #还需要不断地进行状态更新,之后更新状态
        self.robot_vx=0
        self.robot_vy=0
        self.robot_z=0

        #4:开始一个线程,用于接收串口信息,进而更新机器人状态
        Read_thread=threading.Thread(target=self.get_message)
        Read_thread.start()

    def get_send_msg(self,x1=0,x2=0,x3=0,x4=0,x5=0,x6=0,x7=0,x8=0,x9=0,x10=0,x11=0):
        """
        用于串口发送信息,指定功能字与xyz三个信息,进而生成发送信息
        :param function_word:功能字
        :param x: 发送x/vx
        :param y: 发送y/vy
        :param z: 发送z/vz
        :param max_value: 速度中发送的最大速度
        :return:
        """
        #1:确保速度不要太大


        #2:生成数据
        self.messageType.x1=c_ubyte(x1)
        self.messageType.x2=c_ubyte(x2)
        self.messageType.x3=c_ubyte(x3)
        self.messageType.x4=c_ubyte(x4)
        self.messageType.x5=c_ubyte(x5)
        self.messageType.x6=c_ubyte(x6)
        self.messageType.x7=c_ubyte(x7)
        self.messageType.x8=c_ubyte(x8)
        self.messageType.x9=c_ubyte(x9)
        self.messageType.x10=c_ubyte(x10)
        self.messageType.x11=c_ubyte(x11)

        self.send_list=[self.messageType.x1,self.messageType.x2,self.messageType.x3,self.messageType.x4,self.messageType.x5,self.messageType.x6,self.messageType.x7,self.messageType.x8,self.messageType.x9,self.messageType.x10,self.messageType.x11]
        self.send_msg=struct.pack(self.send_type,*self.send_list)
        return self.send_msg

    def get_message(self):
        """
        这里面的逻辑中,没有测试过读取信息较慢的情况下能否正常接收,之后串口出问题很大可能就是这个函数里面出问题
        get_message的函数不断地进行robot_vx,robot_vy,robot_z三个参数的更新,在串口类定义的时候就开启了这个线程
        长数据发送应该是没有大问题的
        :return:
        """
        #1:不断地更新读取的矩阵
        while True:
            #在这里面进行函数的执行
            data=self.USB0.read(100)
            # print("接收到的数据为:",data)
            if len(data)==48:
                analysis_data=struct.unpack('c'*48,data)#解析数据完成
                print(analysis_data)



            time.sleep(0.1)#进行每一轮的休息,避免接收太快出问题

if __name__ == '__main__':
    messageProcesser=MessageProcesser()
    x=500
    while True:
        x=x-1
        if x<0:
            x=500

        # send_message="{},{}/{},{}/{},{}".format(len(place1),big_state_1,len(place2),big_state_2,len(place3),big_state_3)
        send_message=messageProcesser.get_send_msg(x3=int(x),x7=15,x9=21,x10=13,x11=10)


        # print("发送的数据为:",send_message)
        messageProcesser.USB0.write(send_message)#这里面不断地进行发送任务

        time.sleep(0.1)