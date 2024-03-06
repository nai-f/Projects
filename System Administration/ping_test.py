#! /bin/python3

# Naif Alkaltham
# 10/9/2021

import subprocess
import os
import netifaces


select = ""

while select != "quit":
    os.system("clear")

    print("please select from 1-4")
    print("1 - Test connectivity to your gateway")
    print("2 - Test for remote connectivity")
    print("3 - Test for DNS resolution")
    print("4 - Test display gateway IP address")

    select = input(" Please enter a number: ")
    
    gateway = netifaces.gateways()
    default = gateway["default"][netifaces.AF_INET][0]
    
    # The first selection helps to confirm the connection between the client and the gateway
    if select == "1":
                    
        os.system("clear")
        print("Testing your connectivity to the gateway")
        os.system("sleep 3")
        
        check = subprocess.run(["ping", "-c", "1", str(default)], stdout=subprocess.DEVNULL)

        os.system("clear")

        if check.returncode == 0:
            print("the Test was successful")
            os.system("sleep 3")
        else:
            print("the Test faild")
            os.system("sleep 3")

    # the second selcetion helps to confirm the connection between RIT DNS and the client
    elif select == "2":

        os.system("clear")
        print("Testing your remote connectivity")
        os.system("sleep 3")

        check = subprocess.run(["ping", "-c", "1", "129.21.3.17"], stdout=subprocess.DEVNULL)

        os.system("clear")

        if check.returncode == 0:
            print("The Test Was Successful")
            os.system("sleep 3")
        else:
            print("The Test Faild")
            os.system("sleep 3")

    #The third selection helps to confirm the ability to ping using URL
    elif select == "3":

        os.system("clear")
        print("Testing your DNS")
        os.system("sleep 3")

        check = subprocess.run(["ping", "-c", "1", "www.google.com"], stdout=subprocess.DEVNULL)
        os.system("sleep 3")
        os.system("clear")

        if check.returncode == 0:
            print("The Test Was Seccessful")
            os.system("sleep 3")
        else:
            print("The Test Faild")
            os.system("sleep 3")

    # The fourth selection helps to display the Gateway so that the end user doesn't need ro run any commands 
    elif select =="4":

        os.system("clear")
        
        #gateway = os.system("ip r | head -1 | awk '{print $3}'")
        #gateway = netifaces.gateways()
        #default = gateway.get ("default")
        
        #default = gateway["default"][netifaces.AF_INET][0]
        print("Your Gateway IP address is" + str(default))

        os.system("sleep 3")
        os.system("clear")

    # the fifth selection helps to exit from the script with out to do anything else.
    elif select == "Q" or select == "q":
        os.system("clear")
        select = "quit"

    # this statement helps to show that the end user made a wrong selection and let him re-think what are the currect options
    else:

        print("please wait and try again with valid option")
        os.system("sleep 1")
