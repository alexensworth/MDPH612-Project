import pydicom
import matplotlib.pyplot as plt
import numpy as np
import time


###############################################################################
#####        Read the dicom file and extract the needed component         #####
###############################################################################

#Import the files:
df=[]

df1=pydicom.read_file("IM-0019-0001.dcm")
df2=pydicom.read_file("IM-0020-0001.dcm")
df3=pydicom.read_file("IM-0021-0001.dcm")
df4=pydicom.read_file("IM-0022-0001.dcm")

#A figure showing what the original data looks like - these are phantoms. 
#All four figures look roughly the same. It is all the same phantom, the only parameter that changes is the flip angle.



plt.subplot(3, 4, 1)
plt.imshow(df1.pixel_array, cmap=plt.cm.bone)
plt.title('Original Phantom signal 1')
plt.subplot(3, 4, 2)
plt.imshow(df2.pixel_array, cmap=plt.cm.bone)
plt.title('Original Phantom signal 2')
plt.subplot(3, 4, 3)
plt.imshow(df3.pixel_array, cmap=plt.cm.bone)
plt.title('Original Phantom signal 3')
plt.subplot(3, 4, 4)
plt.imshow(df4.pixel_array, cmap=plt.cm.bone)
plt.title('Original Phantom signal 4')




df.append(df1)
df.append(df2)
df.append(df3)
df.append(df4)



#Extract the flip angle and the repitition time

TR=df[0].RepetitionTime

TRtemp=[df[0].RepetitionTime, df[1].RepetitionTime, df[2].RepetitionTime, df[3].RepetitionTime]
for item in TRtemp:
    if TR != item: 
        print("The relaxation times are not all the same. They are:")
        print("Image 1:", df[0].RepetitionTime)
        print("Image 2:", df[1].RepetitionTime)
        print("Image 3:", df[2].RepetitionTime)
        print("Image 4:", df[3].RepetitionTime)
        exit()


print(" ")

FA=[]
for m in range(len(df)):
    FA.append((df[m].FlipAngle)*np.pi/180) #Convert the flip angle from degrees into radians so that np.sin and np.cos can use them properly
    print("Flip angle", m+1, ":", df[m].FlipAngle, "(degrees)")
    print("FA:", FA[m], "(radians)") #Print these next 3 as a sanity check
    print("TR", m+1,":", df[m].RepetitionTime)
    print(" ")
    




###############################################################################
#####                    Manipulate the pixel arrays                      #####
###############################################################################





#Need to convert each of the 2x2 matrices into a vector so that the zipping
#will work properly, will convert back to a 2x2 later
df1_flattened=df1.pixel_array.flatten()
df2_flattened=df2.pixel_array.flatten()
df3_flattened=df3.pixel_array.flatten()
df4_flattened=df4.pixel_array.flatten()


#Data for the non-linear fit function:


#zip the vectors
zipit=zip(df1_flattened, df2_flattened, df3_flattened, df4_flattened)
zipped=list(zipit)




#Remove noisy data that causes the fit function to struggle
for i in range(len(zipped)):
    if sum(zipped[i])<6:
        zipped[i]=(0,0,0,0)
        
        

#Data for the linear fit function:
        
S1y=df1_flattened/np.sin(FA[0])
S2y=df2_flattened/np.sin(FA[1])
S3y=df3_flattened/np.sin(FA[2])
S4y=df4_flattened/np.sin(FA[3])



S1x=df1_flattened/np.tan(FA[0])
S2x=df2_flattened/np.tan(FA[1])
S3x=df3_flattened/np.tan(FA[2])
S4x=df4_flattened/np.tan(FA[3])



zipity=zip(S1y,S2y,S3y,S4y)
zippedy=list(zipity)

zipitx=zip(S1x,S2x,S3x,S4x)
zippedx=list(zipitx)





###############################################################################
#####                           Curve fitting                             #####
###############################################################################

#Now we are ready to curve fit the data
from scipy import optimize

def Non_lin_func(x, M, E):
    return ( M * (1-E) * np.sin(x) ) / ( 1 - ( E * np.cos(x) ) )


def Lin_func(x, E, M):
    return E*x + M*(1-E)



#Parameters for the non-linear fit:
E_param=[]
M_param=[]
#Parameters for the linear fit:
Slope_param=[]
Inter_param=[]

#len(zipped) = 192*192= 36864
#Go through each of the zipped data, which is now a list, where each element has 4 values.. For example:
#zipped[0]=(0,0,0,0)
#zipped[1]=(0,0,0,0)
#zipped[11000]=(1,2,0,1) -> This will get converted to (0,0,0,0), as the fit function will struggle to fit this. The signal is not strong enough to give reliable data.
#zipped[25808]=(31,70,100,101)


#It will yield the parameters M, which is the initial magnetization and E, which is the 
#exponential decay equation that relates the desired quantity, T1 to the relaxation time, TR.


#The code that actually does the fitting (non-linear fit):

print("Starting the non-linear fit:")
start_non_lin=time.time()
for i in range(len(zipped)):
    params, params_covariance = optimize.curve_fit(Non_lin_func, FA, zipped[i], p0=[3,0.9], maxfev=10000, bounds=[[0, 0], [1350,0.99]])
    M_param.append(params[0])
    E_param.append(params[1])
    if i % 1000 == 0:
        print("Fit completed for:", i,"/", len(zipped), "pixels.") #Update the user so they have an idea of how long it is taking.
end_non_lin=time.time()

"""
#A secondary fitting algorithm.
#This makes the code run faster (it skips fitting over the all zero values and gives them predetermined values), however it might not give the most accurate and reliable results..
for i in range(len(zipped)):
    if all([v==0 for v in zipped[i]]):
        E_param.append(0.96)
        M_param.append(0)
    else:
        params, params_covariance = optimize.curve_fit(Non_lin_func, FA, zipped[i], p0=[3,0.96], maxfev=10000, bounds=[[0, 0], [1350,1]])
        M_param.append(params[0])
        E_param.append(params[1])
    if i % 1000 == 0:
        print(i)
"""

#print("Checkpoint 3.0")


#On to linear fitting
print("Non-linear fit complete.")
print("")
print("Starting the linear fit:")
start_lin=time.time()
for i in range(len(zippedy)):
    params, params_covariance = optimize.curve_fit(Lin_func, zippedx[i], zippedy[i], p0=[0.9,0.9], maxfev=10000, bounds=[[0, 0], [0.99,1350]])
    Slope_param.append(params[0])
    Inter_param.append(params[1])
    if i % 1000 == 0:
        print("Fit completed for:", i,"/", len(zipped), "pixels.")
end_lin=time.time()
print("Linear fit complete.")







###############################################################################
#####                    Data manipulation, find T1                       #####
###############################################################################




####Non-Linear####

#Converts E into T1:

#Ensure the array is a numpy array, not a list.
array_of_E=np.asarray(E_param)


#This for loop just goes through the results and reduces very large values of E to 0.99. 
#Values would be so large that they would distort the range of the resulting T1 image, 
#removing the ability to see the image's fine detail.
for j in range(len(zipped)):
    if array_of_E[j]>0.99:
        array_of_E[j]=0.99

#Change the 1D vector back into a 192x192 matrix
E_mat = np.array(array_of_E).reshape(192,192) 
#Create a 192x192 matrix of T1 values
T1=-1*TR/np.log(E_mat)

#This gives you a magnetization image
array_of_M=np.asarray(M_param)
M_mat = np.array(array_of_M).reshape(192,192)


#####Linear#####

array_of_Slope=np.asarray(Slope_param)
Slope_mat = np.array(array_of_Slope).reshape(192,192)
T1_lin=-1*TR/np.log(Slope_mat)







###############################################################################
#####                        Plotting the Data                            #####
###############################################################################





#plot the data

####Non-linear:####

#plot the magnetization parameter
plt.subplot(3, 4, 6)
plt.imshow(M_mat, cmap=plt.cm.bone)
plt.title('M0 parameter')

#plot the E1 parameter
plt.subplot(3, 4, 7)
plt.imshow(E_mat, cmap=plt.cm.bone)
plt.title('E1 parameter')


#Plot the T1 parameter
plt.subplot(3, 4, 8)
plt.imshow(T1, cmap=plt.cm.bone)
plt.title('Converted to T1')




####Linear:####

#plot the Slope/E1 parameter
plt.subplot(3, 4, 11)
plt.imshow(Slope_mat, cmap=plt.cm.bone)
plt.title('Slope/E1 parameter')

#plot the converted T1 parameter
plt.subplot(3, 4, 12)    
plt.imshow(T1_lin, cmap=plt.cm.bone)
plt.title('Converted to T1')




#print("Checkpoint 3.1")

print("The non-linear fit took:", end_non_lin-start_non_lin, "seconds")
print("The linear fit took:", end_lin-start_lin, "seconds")





###############################################################################
#####         Demonstrate the plotting for single sets of data            #####
###############################################################################




#Uncomment this and can change the fitting data to be anything you wish from zipped[0] to zipped[36863]. Some examples: 
#zipped[0]=(0,0,0,0)
#zipped[1]=(0,0,0,0)
#zipped[11000]=(1,2,0,1)                (random non-zero value)
#zipped[25808]=(31,70,100,101)          (very close to maximum value)
    


#Pick the pixel you wish to plot:
pixel=25808



####Non-Linear Function####

params, params_covariance = optimize.curve_fit(Non_lin_func, FA, zipped[pixel], p0=[3,0.9], maxfev=10000, bounds=[[0, 0], [1350,0.99]])
E_param1=params[1]
M_param1=params[0]

print(" ")
print("Results of single dataset fit:")
print("Non-linear fit:")
print("E1 parameter for pixel number", pixel, ":", E_param1)
print("M0 parameter for pixel number", pixel, ":", M_param1)
T1_=-1*TR/np.log(E_param1)
print("T1 parameter for pixel number", pixel, ":", T1_)

#print("Checkpoint 3.2")

#Plot the non-linear fit
plt.subplot(3, 4, 5)
plt.scatter(FA, zipped[pixel], label='Non-linear Data')
plt.plot(FA, Non_lin_func(FA, params[0], params[1]), label='Non-linear fitted function')
plt.xlabel('Flip angle (radians)')
plt.ylabel('Signal intensity')
plt.title('Non-linear fit')





####Linear function####

params, params_covariance = optimize.curve_fit(Lin_func, zippedx[pixel], zippedy[pixel], p0=[0.9,0.9], maxfev=10000, bounds=[[0, 0], [0.99,1350]])
Slope_par=params[0]
Inter_par=params[1]
  
Slope_param1=np.asarray(Slope_par)
Inter_param1=np.asarray(Inter_par)

print(" ")
print("Results of single dataset fit:")
print("Linear fit:")
print("Slope parameter (E1) for pixel number", pixel, ":", Slope_param1)
print("M0 parameter for pixel number", pixel, ":", Inter_param1)
T1_lin=-1*TR/np.log(params[0])
print("T1 paramter for pixel number", pixel, ":", T1_lin)


#print("Checkpoint 3.2")

plt.subplot(3, 4, 9)
plt.scatter(zippedx[pixel], zippedy[pixel], label='Linear data')
plt.plot(zippedx, Lin_func(zippedx, Slope_param1, Inter_param1), label='Linear fitted function')
plt.xlabel('S/tan(\u03B1)')
plt.ylabel('S/sin(\u03B1)')
plt.title('Linear fit')




plt.show()

