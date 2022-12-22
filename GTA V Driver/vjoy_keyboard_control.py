import pyvjoy
import win32gui, win32api

j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768

while 1:
	
    w_pressed = int(win32api.GetAsyncKeyState(0x57)&0x8001 > 0)
    a_pressed = int(win32api.GetAsyncKeyState(0x41)&0x8001 > 0)
    s_pressed = int(win32api.GetAsyncKeyState(0x53)&0x8001 > 0)
    d_pressed = int(win32api.GetAsyncKeyState(0x44)&0x8001 > 0)
   
    #print(w_pressed)
   	
    j.data.wAxisX = int((1 - a_pressed + d_pressed)/2 * vjoy_max)
    j.data.wAxisY = int(w_pressed * vjoy_max)
    j.data.wAxisZ = int(s_pressed * vjoy_max)
    
    #j.data.wAxisY = int(0.5 * vjoy_max)
    
    print(j.data.wAxisX, j.data.wAxisY, j.data.wAxisZ)
    
    j.update()
    


j.data.wAxisX = int(1 * vjoy_max)
j.update()


plt.hist((np.arange(-1,1, 0.0001)+1)/2)
