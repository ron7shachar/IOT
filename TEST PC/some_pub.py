from sys import path as sysPath
import rticonnextdds_connector as rti
from os import path as osPath
from time import sleep
import random
filepath = osPath.dirname(osPath.realpath(__file__))

connector = rti.Connector("MyParticipantLibrary::Example_pub", filepath + "/DDS.xml")
outputDDS = connector.getOutput("MyPublisher::MyWriter")

while True:
    randomNumb = random.randint(0, 99)
    status = '[MB-OK]'
    outputDDS.instance.setNumber("NumberMember", randomNumb)
    outputDDS.instance.setString("StingMember", status)
    outputDDS.write()
    sleep(11)
    print(f'published: {randomNumb} ')
    print(f'published-MB:  Status: {status}')
    print(f'published: {randomNumb} ,Status: {status}')

