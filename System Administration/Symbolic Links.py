#! /bin/python3
# Naif Alkaltham 
#/31/10/2021
import os
import subprocess 

select = ""

while select != "quit":
    os.system("clear")



    print("1 - create a shortcut in your home directory.")
    print("2 - Remove a shortcut from your hoem directory.")
    print("3 - Run shortcut report.")

    select = input("please select input choice or quit to quit: ")
    


    #find the username
    USERNAME = (subprocess.check_output("whoami", shell=True, universal_newlines=True)).strip()

    #userdir
    USERDIR = "/home/" + USERNAME

    

    if select == "1":
        os.system("clear")

        #file source 
        filename = input("please enter file name to create a shortcut or R/r to return: ")

        if(filename == "R" or filename == "r"):
            print("Returing")
            os.system ("sleep 2")
        elif (filename == "" or filename == " "):
            print("Wrong input")
            print("Returing")
            os.system("sleep 2")

        else:
            
            
            #searchfile
            filelocation = (subprocess.check_output("find "+" "+ USERDIR  +" -name"+" "+filename, shell=True, universal_newlines=True)).strip()


         # the file exits 
            try:

                if filelocation != "":
                    
                    print("Found " + filelocation)
                    
                    os.system("ln -s "+ filelocation +" "+ USERDIR)
                    print("the shortcut has been created")
                    print("Returing")
                    os.system("sleep 1")
                
                elif (filelocation == "r" or filelocation == "R"):
                    os.system ("clear ")
                    print("Returing")
                    os.system ("sleep 1")
                                        
                
                else:
                    print("the file doesn't exists")
                    print("Returing")
                    os.system("sleep 2")
            except:
                print("the file does exists!!!")
                print("Returing")
                os.system("sleep 2")


        #back1 = input("To return to the Main Menu, Press Enter ")


    elif (select == "2"):
        os.system("clear")
        
        #back2 = input("Please Press Enter To return to the Main Menu, or R/r ro remove a link ")

        
        remove = input("please enter the shorcut/link to remove:   ")
        try:
            use = (subprocess.check_output("unlink "+USERDIR+"/"+remove, shell=True, universal_newlines=True)).strip()
    
            if use == " " or use == "":
                print("the file has been removed ")
                print("Returing")
                os.system("sleep 2")

            else:
                print("the file doesn't exist")
                print("returing")
                os.system("sleep 2")
        except:
            print("the file doesn;t exist ")
            print("Returing")

        os.system("sleep 3")
        





    elif (select == "3"):
        os.system("clear")

        current = (subprocess.check_output("pwd", shell=True, universal_newlines=True)).strip()

        print("Your current directory is: "+ current)

        symlinks = (subprocess.check_output("ls -la "+" "+ USERDIR  +""" | grep "\->" """+  " | awk '{print $9}'" , shell=True, universal_newlines=True)).strip()
        originals = (subprocess.check_output("ls -la "+" "+ USERDIR  +""" | grep "\->" """+  " | awk '{print $11}'" , shell=True, universal_newlines=True)).strip()
        count =  (subprocess.check_output("ls -la "+" "+ USERDIR  +""" | grep "\->" """+  " | awk '{print $11}'" + " | wc -l", shell=True, universal_newlines=True)).strip()

        print("The number of links is: " + count +'\n')

        print("Symbolic\n"+symlinks.strip()+" "+"\nOriginals\n"+originals.strip() )
        #print(symlinks.strip() +"                      "+ originals.strip() ) 
        
        #table_data = [ [ "symlinks", symlinks], ["originals", originals], ]
        #print(table_data)
        #for row in table_data:
           # print("{: >1} {: >1}".format(*row))

        back3 = input("Please Press Enter To return to the Main Menu:  ")

        if back3 == " " or back3 == "":
            os.system("sleep 0")
        elif (back3 == "r" or back3 == "R"):
            os.system("sleep 0")
            os.system("clear")
            select == "2"
            
        else:
            print("please chioce correct keys")


    elif select == "Q" or select == "q":
        os.system("clear")
        select == "quit"

    else:
        os.system("clear")
        print("Please Enter Valid option")
        os.system("clear")

