import os

#create a vm 
def create():
    
    name = input("Please specify the name of the new VM: ")
    operating_system = input(f"Please specify the Operating System of {name}: ")
    path = input("Please specify the path to " + operating_system + " : ")
    ram = input("Please specify the RAM that would you like " + name + " to have: ")
    os.system('vboxmanage createvm --name ' + name + ' --ostype ' + operating_system + ' --register')
    os.system('vboxmanage storagectl ' + name + ' --name IDE --add ide')
    os.system('vboxmanage modifyvm ' + name + ' --memory ' + ram)
    os.system(
        'vboxmanage storageattach ' + name + ' --storagectl IDE --port 0 --device 0 --type dvddrive --medium ' + path)

#list the vms
def show_options():
    
    var = input('Would you like detailed list? [y/n]: ')
    while var not in 'yn':
        var = input("Please enter [y/n]: ")
    if var == 'y':
        os.system('vboxmanage list vms --long')
    elif var == 'n':
        os.system('vboxmanage list vms')

#start the vm
def start(vm_name):
    
    os.system('vboxmanage startvm ' + vm_name)

#stop the vm
def stop(vm_name):
    
    os.system('vboxmanage controlvm ' + vm_name + ' poweroff soft')

#show the vm settings
def show_settings(vm_name):
    
    os.system('vboxmanage showvminfo ' + vm_name)

#delete the vm
def delete(vm_name):
   
    os.system('vboxmanage unregistervm ' + vm_name + ' --delete')

# The main menu 
def main():
    
    option = ''
    while option != 'Q':
        print('\n************************ \n'
              '>1. : Create VM        < \n'
              '>2. : List all VMs     < \n'
              '>3. : Start VM         < \n'
              '>4. : Stop VM          < \n'
              '>5. : Show VM Settings < \n'
              '>6. : Delete VM        < \n'
              '>Q. : Quit             < \n'
              '************************')

        option = input("Please select an options: ")

        if option == '1':
            create()

        elif option == '2':
            show_options()

        elif option == '3':
            vm_name = input('Please specify the VM name: ')
            start(vm_name)

        elif option == '4':
            vm_name = input('Please specify the VM name: ')
            stop(vm_name)

        elif option == '5':
            vm_name = input('Please specify the VM name: ')
            show_settings(vm_name)

        elif option == '6':
            vm_name = input('Please specify the VM name: ')
            delete(vm_name)
        elif option == 'Q' or option == 'q':
            break
        else:
            print("\nOption entered is INVALID, please try again.")


main()


