import numpy as np
import cv2
from cv2 import cv
import matplotlib.pyplot as plt
import pickle

'''Returns a numpy array cooresponding to the images in the file '''
def load_images(filename,num_images):
	x = np.fromfile(filename, dtype='uint8')
	x = x.reshape((num_images,60,60))
	return x


'''Given a threshold and an image, return the imaeg where all pixels with value > threhold are set to max(255), and < threshold are set to min(0)'''
def threshold(image,threshold):
	new = image.copy()
	for i in range(0,len(new)):
		for j in range(0,len(new[i])):
			if new[i][j] < threshold:
				new[i][j] = 0
			else:
				new[i][j] = 255
	return new

def smooth(image):
	mat = cv.fromarray(image)
	dest = cv.fromarray(np.zeros((len(image),len(image)),np.uint8))
	#GaussianSmooth
	#cv.Smooth(mat,dest)
	cv.Smooth(mat,dest,smoothtype=cv.CV_MEDIAN)
	#cv.Set(dest)


	return np.asarray(dest[:][:],np.uint8)
'''https://en.wikipedia.org/wiki/Unsharp_masking'''
def sharpen(image):
	#using a basic kernel to sharpen images
	kernel = np.array([[0,0,0],[0,3,0],[0,0,0]],np.uint8)
	dest = cv.fromarray(np.zeros((len(image),len(image)),np.uint8))
	dest = cv2.filter2D(image,-1,kernel)

	return np.asarray(dest[:][:],np.uint8)

'''Show the images in the list provided'''
def show_images(image_list, num_images):
	if num_images > 7:
		num_images = 7
	fig = plt.figure()
	for i in range(0,num_images):
		a= fig.add_subplot(1,num_images,i+1)
		plt.imshow(image_list[i],cmap='Greys_r')
	plt.show()

def edge_detect(image):
	mat = cv.fromarray(image)
	dest = cv.fromarray(np.zeros((len(image),len(image)),np.uint8))

	dest = cv2.Sobel(image, -1,1,1)


	return np.asarray(dest[:][:],np.uint8)

def reverse(image):
	new = image.copy()
	for i in range(0,len(new)):
		for j in range(0,len(new[i])):
			new[i][j] = 255 - new[i][j]
	return new
def erode(image):
	mat = np.asarray(image)
	kernel  = np.ones((2,2),np.uint8)
	dest = cv2.erode(mat, kernel,iterations = 1)
	return dest

def dilate(image):
	mat =np.asarray(image)
	kernel  = np.ones((3,3),np.uint8)
	kernel[0][0] = 0
	kernel [0][2] = 0
	kernel[2][0] = 0
	kernel[2][2] = 0
	dest = cv2.dilate(mat, kernel,iterations = 1)
	return dest

'''Returns a post processed image.'''
def process_image(image):
		
	b = threshold(image,250)
	# b_2 =erode(b)
	b_3 = dilate(b)
	b_3 = erode(b_3)

	return b_3

def process_and_save(images):
	for i in range(0,len(images)):
		images[i] = process_image(images[i])
	
	print "Processed"	
	with open('processed.pkl','w') as f:
		pickle.dump(images,f)
	
	print "Saved full images"
	
	with open('split.pkl','w') as f:
		to_save = process_all(images)
		print "split"
		pickle.dump(to_save,f)



'''If a contour is believed to have more than one within it then the contour is split into two. '''
def split_contour(image, box):
	[x,y,w,h] = box
	new_img = image.copy()
	#show_images([new_img[y:y+h][x:x+w]],1)
	if (w>=h):
		#print "w"+ str([x,y,w,h])
		if(w>28):
			split = 0
			while(split < w/2-5):
			# for i in range(x+5,x+w-5):
				for i in [x+w/2 - split, x+w/2 + split] :
					#new_img = image.copy()
					for a in range(y,y+h):
						for b in range(i-1,i+1):
							new_img[a][b] = 0

					#show_images([new_img],1)
					contours, _ = cv2.findContours(new_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
					num_contours = 0
					for contour in contours:
						[x_c,y_c,w_c,h_c] = cv2.boundingRect(contour)
						if (w_c> 2 and h_c>7) and (w_c<28 and h_c < 28):
							num_contours+=1
					if num_contours == 2:
						#print "Found New contours"
						return contours
				split +=1
		if(h>28):
			#print "H" +  str([x,y,w,h])
			split = 0
			while(split < h/2-5):
			# for i in range(x+5,x+w-5):
				for j in [y+h/2 - split, y+h/2 + split] :
			# for j in range(y+5,y+h-5):
					#new_img[j-1:j+1][x:x+w] = 0
					new_img = image.copy()
					for a in range(j-1,j+1):
						for b in range(x,x+w):
							new_img[a][b] = 0
					#show_images([new_img],1)
					contours, _ = cv2.findContours(new_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
					num_contours = 0
					for contour in contours:
						[x_c,y_c,w_c,h_c] = cv2.boundingRect(contour)
						if (w_c> 2 and h_c>7) and (w_c<28 and h_c < 28):
							num_contours+=1
					if num_contours == 2:
						#print "Found new Contours"
						return contours
				split += 1

	else:

		
		if(h>28):
			#print "H" +  str([x,y,w,h])
			split = 0
			while(split < h/2-5):
			#	print "AAA"

			# for i in range(x+5,x+w-5):
				for j in [y+h/2 - split, y+h/2 + split]:
			# for j in range(y+5,y+h-5):
					#new_img[j-1:j+1][x:x+w] = 0
					new_img = image.copy()
					for a in range(j-1,j+1):
						for b in range(x,x+w):
							new_img[a][b] = 0
				#	print "BBB"
					#show_images([new_img],1)
					contours, _ = cv2.findContours(new_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
					num_contours = 0
					for contour in contours:
						[x_c,y_c,w_c,h_c] = cv2.boundingRect(contour)
						if (w_c> 2 and h_c>7) and (w_c<28 and h_c < 28):
							num_contours+=1
					if num_contours == 2:
					#	print "Found new Contours"
						return contours
				split +=1

		if(w>28):
			split = 0
			while(split < w/2-5):
			# for i in range(x+5,x+w-5):
			#	print "CCC"
				for i in [x+w/2 - split, x+w/2 + split] :
					new_img = image.copy()
					#new_img[y:y+h][i-1:i+1] = 0
					for a in range(y,y+h):
						for b in range(i-1,i+1):
							new_img[a][b] = 0
			#		print "DDD"
					#show_images([new_img],1)
					contours, _ = cv2.findContours(new_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
					num_contours = 0
					for contour in contours:
						[x_c,y_c,w_c,h_c] = cv2.boundingRect(contour)
						if (w_c> 2 and h_c>7) and (w_c<28 and h_c < 28):
							num_contours+=1
					if num_contours == 2:
					#	print "Found New contours"
						return contours
				split += 1

	return []
''' Find possible locations of digits within the image'''
def getContours(image):
	digits = []
	# contours, _ = cv2.findContours(image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	contours, _ = cv2.findContours(image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	num_contours = 0
	split = 0
	#Count the number of contours
	for contour in contours:

		[x,y,w,h] = cv2.boundingRect(contour)
		#print x,y,w,h
		if (w> 1 and h>7) and (w<28 and h < 28):
			num_contours+=1

	#If there aren't enough then split the image to try and find contours
	if(num_contours==1):
		contours = split_contour(image.copy(),[x,y,w,h])
		split = 1
	elif(num_contours==0):
		contours = split_contour(image.copy(),[0,0,60,60])
		split = 1

	#Return two images with the contours found centered within the image. 
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)


		if (w> 2 and h>7) and (w<28 and h < 28):
			# print x,y,w,h
			# print x+w
			# print y+h
			dig = np.zeros((28,28),np.uint8)
			s_x = (28-w)/2
			s_y = (28-h)/2
			for j in range(y,y+h):
				for i in range(x,x+w):
					dig[s_y+(j-y)][s_x+(i-x)] = image[j][i]
			#show_images([dig],1)
			digits.append(dig.copy());

		elif(w>28 or y>28):

			new_contours = split_contour(image.copy(),[x,y,w,h])
			split =1 
			if len(new_contours) > 0:
				contours = contours + new_contours

			
	#if split == 1:
	return digits, split 	
	#return digits, split

#Given a list of contours find the 2 best. 
def find_best(ls):
	dens = []
	sum = 0
	for i in range(0,len(ls)):
		sum = 0
		for y in range(0,len(ls[i])):
			for x in range(0,len(ls[i][y])):
				sum += ls[i][y][x]
		dens.append([sum/(28.0*28.0),i])

	#print dens
	dens.sort()
	#print dens
	return [ls[dens[-1][1]],ls[dens[-2][1]]]
		
'''Takes un processed images and returns a list of tuples(image,image) cooresponding to the 2 digits within the unprocessed image.'''		
def process_all(images):
	ret_list = []
	for a in images:
		a_new = process_image(a)

		seperated, split = getContours(a_new)
		if(len(seperated)> 2):
			seperated = find_best(seperated)

		if(len(seperated)==1):
			seperated = seperated + seperated


		if(len(seperated) == 0):
			seperated = [np.zeros((28,28),np.uint8)]*2

		ret_list.append(seperated)
	return ret_list



if __name__ == "__main__":
	#Example usage:

	imgs = load_images('train_x.bin',100000)
	process_and_save(imgs)
	#pairs = process_all(imgs)
	
	# #process_and_save(imgs)
	# failures  = 0
	# total = 0
	# num_splits = 0
	# for a in imgs[0:1000]:
	# #a= imgs[0]
	# #print a
	# 	if total % 1000 == 0 :
	# 		print total
	# 	total += 1
	# 	a_new = process_image(a)
	# #show_images([a,a_new],2)
	# 	seperated, split = getContours(a_new)
	# 	if(len(seperated)> 2):
	# 		#failures += 1
	# 		seperated = find_best(seperated)
	# 		#show_images([a,a_new]+seperated,2+len(seperated))
	# 	if(len(seperated)==1):
	# 		seperated = seperated + seperated
	# 		#show_images([a,a_new]+seperated,2+len(seperated))
	# 		failures += 1

	# 	if(len(seperated) == 0):
	# 		seperated = [np.zeros((28,28),np.uint8)]*2
	# 		failures += 1
	# 		#show_images([a,a_new]+seperated,2+len(seperated))
	# 		#print split
	# 	#if split ==1:
	# 		#show_images([a,a_new]+seperated,2+len(seperated))
		
	# 	num_splits += split
	# print failures/float(total)
	# print num_splits/float(total)

