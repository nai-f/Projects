#!/usr/bin/python3
#Naif Alkaltham
#11/14/2020

import os
import subprocess

from geoip import geolite2

ips = []
valid_counts = []
#countlist =  []
def getips():
    addresseslist = list(set((subprocess.check_output("""grep "Failed password" /home/student/Scripts/Script4/syslog.log | awk '{print $11}' """ , shell=True, universal_newlines=True)).split("\n")))
    
    for i in addresseslist:
        if ("." in i):
            ip = i.strip().split(".")
            if (len(ip) == 4):
                for x in ip:
                    if (x.isnumeric() and int(x)>=0 and int(x) <=255):
                        ips.append(i)
                        
   

def remove_duplicate(old_list):
    new_list = set(old_list)
    return new_list


def cout(ip_address):
    countlist = list((subprocess.check_output("grep " +""" "Failed password for root from """+ ip_address+"""" """+ "/home/student/Scripts/Script4/syslog.log", shell=True, universal_newlines=True)).split("\n"))
    count = 0
    for i in countlist:
        if i != "":

            count += 1

    return count 


def getlocation(ip_address):
    location = geolite2.lookup(ip_address)
    return location.country

def script4():
    os.system("clear")
    date= (subprocess.check_output("date" , shell=True, universal_newlines=True))
    print("Attacker Report - "+ date)
    print("COUNT"+"   "+"IP"+"              "+"Country")
    for i in valid_ips:
        #print(getlocation(i))
        #print(cout(i), str(i), getlocation(i), sep="     ")
        print("{:<7} {:<15} {:<4}".format(cout(i),str(i),getlocation(i)))

getips()
valid_ips = remove_duplicate(ips)
script4()

#print(valid_ips)


#countlist = list((subprocess.check_output("grep " +""" "Failed password for root from 159.122.220.20" """+ "/home/student/Scripts/Script4/syslog.log", shell=True, universal_newlines=True)).split("\n"))
#print(countlist)
