## python3 kmeans.py --dataset dataset --respath KM --clusters 5

# import the necessary packages
from sklearn.cluster import KMeans # scikit-learn implementation of k-means
import argparse #To parse command line arguments
import utils #to use helper functions
import cv2 #openCv bindings
import glob #grabbing the file paths to our images

# construct the argument parser and parse the arguments
"""
The following Code parses our command line arguments.
--dataset - the file path to our images
--espath - the path where the results will be saved and
--clusters, the number of clusters that we wish to generate.
"""
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-r", "--respath", required = True,
	help = "Path to the result path")
ap.add_argument("-c", "--clusters", required=True, type=int,
				help="# of clusters")
args = vars(ap.parse_args())

# ==========================
def getimages():

	# use glob to grab the image paths and loop over them
	for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		imageID = imagePath[imagePath.rfind("/") + 1:]
		clean = km(imagePath)
		cv2.imwrite(args["respath"] + "/" + imageID, clean)
def km(path):
	img = cv2.imread(path)
	# reshape the image to be a list of pixels
	image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	image = img.reshape((image.shape[0] * image.shape[1], 3))

	# cluster the pixel intensities
	clt = KMeans(n_clusters=args["clusters"])
	clt.fit(image)
	
	#count the number of pixels that are assigned to each cluster
	hist = utils.centroid_histogram(clt)
	# the figure that visualizes the number of pixels assigned to each cluster
	bar = utils.plot_colors(hist, clt.cluster_centers_)
	
	return bar
#===========================

getimages()