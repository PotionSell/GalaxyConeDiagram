#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:46:04 2019

@author: bscousin

BACKUP BEFORE MOVING TO A "SECTOR" CLASS FOR ALL MY PREVIOUSLY-CONVOLUTED USAGE OF SECTORS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Wedge
from matplotlib.ticker import FuncFormatter   #for displaying polar angles in decimals

from astropy import units as u
from astropy.coordinates import SkyCoord, Distance, Angle, IllegalHourError
from astropy.constants import c
from astropy.cosmology import Planck15

from scipy.spatial import cKDTree

from os.path import splitext, join



def transformRadius(r, rMin, rMax):
    '''
    Transforms a real coordinate radius into Matplotlib's polar axis system,
    which follows an internal coord system instead of a real/physical one: the
    polar plot is always centered on (0.5,0.5) and is of radius 0.5.

    When plotting points onto the polar plot, their coords are transformed
    internally to display properly. But when plotting anything else (like a
    patch artist), you must manually transform the coordinates.

    This function transforms a radius coord into the polar plot's scaled
    coordinates. (I would've made this a lambda function, but I wanted to
    document this silly Matplotlib nuance....)

    r:       the radius you wish to transform
    rMin:    minimum radius of the plot axis
    rMax:    maximum radius of the plot axis
    '''
    #transformRadius = lambda r,rMin,rMax: 0.5 * (r-rMin)/(rMax-rMin)
    return 0.5 * (r-rMin)/(rMax-rMin)

#function for formatting the polar tickmarks into decimal degrees from radians
rad2fmt = lambda x,tickPos: f'{np.rad2deg(x):.2f}°'


#####helper functions for the draw_wedge() function of the coneData class#####
##see for explanation: https://stackoverflow.com/a/54400045

def perp( a ) :
    ##from https://stackoverflow.com/a/3252222/2454357
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seq_intersect(a1,a2, b1,b2) :
    ##from https://stackoverflow.com/a/3252222/2454357
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def angle(a1, a2, b1, b2):
    ##from https://stackoverflow.com/a/16544330/2454357
    x1, y1 = a2-a1
    x2, y2 = b2-b1
    dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # determinant
    return np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

###########################################################################




class coneData():


    def __init__(self, datapath, fnames, fieldName='', fieldCenter=[26.16670584,27.90981659]):
        #note: I'm assuming through this class that /datapath contains both the
        #SDSS data and the other data.
        self.datapath = datapath
        self.fnames = fnames
        #self.fnameSURVEYS = fnameSURVEYS
        self.numDatasets = len(fnames)

        #self.SURVEYdata = [None]*self.numSURVEYS
        #self.goodSURVEYdata = [None]*self.numSURVEYS
        self.fieldName = fieldName
        self.fieldCenter = fieldCenter
        self.cleanData = [None]*self.numDatasets
        self.dataCoords = [None]*self.numDatasets

    def getCenter(self,thetaVal):
        if thetaVal == 'RA':
            return self.fieldCenter[0]
        else:
            return self.fieldCenter[1]

    def loadDatasets(self):
        '''
        Load in the field's datasets given their file names.

        I ASSUME THAT THE SDSS DATASET IS THE FIRST ONE THAT IS LOADED...
        and then SFACT, then SWB's data.
        '''
        data = [None]*self.numDatasets

        for i in range(self.numDatasets):
            fname = self.fnames[i]
            fpath = join(self.datapath, fname)

            #load SDSS data first
            if i == 0:
                loadedData = pd.read_csv(fpath, header=1)
            else:
                loadedData = pd.read_csv(fpath, sep='\s+', header=1)
            data[i] = loadedData
        self.data = data


    def cleanDatasets(self):
        '''
        Filters/masks galaxy redshift/coordinate data based on certain flags.
        '''

        nullVal = -9.99990
        nullStr = 'INDEF'
        nullType = 17

        RAfield = 'RA'
        DECfield = 'DEC'
        Zfield = 'Z_EST'

        typefield = 'ELGTYPE'

#        SURVEYdata = self.data[1:]

        for i in range(self.numDatasets):

            #clean only the SFACT data, assuming that the SDSS and SWB data is clean...
            if i == 1:
                data = self.data[i]
                #ensure that the data is in floats, not strings of numbers......
                data[Zfield] = pd.to_numeric(data[Zfield], errors='coerce')
                data[typefield] = pd.to_numeric(data[typefield], errors='coerce', downcast='integer')
                #only 'INDEF' fields will be coerced into NaNs, so drop them:
                data = data.dropna(subset = [typefield])

                #then, drop rows that have null values for redshift or are marked as a wrong detection
                cleanData = data[ (data[Zfield] != nullVal) & (data[typefield] != nullType)]

            else:
                cleanData = self.data[i]

            self.cleanData[i] = cleanData


    def angleConv(self,thetaVals,dataFormat):
        '''
        Convert an arbitrary angle (e.g., either RA or Dec) thetaVals to radians
        based on its data file format.

        dataFormat:
            SDSS    both RA and Dec are in degrees
            SWB     both RA and Dec are in degress
            SFACT   RA is in hours; Dec is in degrees
        '''

        if (dataFormat == 'SDSS') or (dataFormat == 'SWB'):
            #SDSS and SWB's data has RA/DEC in degrees
            thetaDeg = Angle(thetaVals, u.degree)
            thetaRad = thetaDeg.radian
        elif dataFormat == 'SFACT':
            #SFACT has RA in hours, DEC in degrees
            #try RA hour conversion first; otherwise, it's DEC in degrees
            try:
                thetaHours = Angle(thetaVals, u.hour)
                thetaRad = thetaHours.radian
            except IllegalHourError:
                thetaDegs = Angle(thetaVals, u.degree)
                thetaRad = thetaDegs.radian
        else:
            raise ValueError('Please enter a value data format (e.g., \'SDSS\'')
        return thetaRad

    def loadCoords(self):
        '''
        ??

        Careful... this requires that the data (SFACT particularly) is cleaned
        so that any odd values (like strings for RA) are removed.
        '''
        for i in range(self.numDatasets):
            data = self.cleanData[i]

            #SDSS data format
            if i == 0:
                RAfield = 'ra'
                DECfield = 'dec'
                Zfield = 'redshift'
                dataFormat = 'SDSS'
            #SFACT format
            if i == 1:
                RAfield = 'RA'
                DECfield = 'DEC'
                Zfield = 'Z_EST'
                dataFormat = 'SFACT'
            #Sam format
            if i > 1:
                #RAfield = 'RA(H:M:S)'
                RAfield = 'RA(deg)'
                DECfield = 'Dec(deg)'
                Zfield = 'z'
                dataFormat = 'SWB'

            RAs = data[RAfield]
            DECs = data[DECfield]
            Zs = data[Zfield]

            RAs = self.angleConv(RAs,dataFormat)
            DECs = self.angleConv(DECs,dataFormat)

            self.dataCoords[i] = (RAs,DECs,Zs)


    def coneDiagram(self, thetaVal='RA', minTheta=None, maxTheta=None, minZ=0, maxZ=2, rOrigin=0, overplot=False):

        colors = ['k','g','b','r']
        markers = ['o','*','*','*']
        sizes = [1,8,8,8]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')

        for i in range(self.numDatasets):

            (RAs,DECs,Zs) = self.dataCoords[i]

            if thetaVal=='RA':
                thetaRad = RAs
            elif thetaVal=='Dec':
                thetaRad = DECs
            else:
                raise ValueError('Specify a valid coordinate for theta (RA or DEC)')

            ax.scatter(thetaRad,Zs, c=colors[i], marker=markers[i], s = sizes[i])
            #note: this is NOT plotting each point individually; this plots
            #each dataset individually

        #ax.set_rmax(maxZ)
        #ax.set_rmin(minZ)
        ax.set_ylim([minZ,maxZ])
        ax.set_rorigin(rOrigin)

        #set RA limits to be scaled from the dataset's own limits
        #note: this will be from the SDSS dataset since it has the largest range
        if minTheta is None:
            minTheta = 0.9*min(thetaRad)

        if maxTheta is None:
            maxTheta = 1.1*max(thetaRad)
            if maxTheta > 2*np.pi: maxTheta = 2*np.pi

        ticks = np.linspace(minTheta, maxTheta, 4)
        ax.set_xticks(ticks)

        #convert to decimal precision
        ax.xaxis.set_major_formatter(FuncFormatter(rad2fmt))

        #you must call these AFTER setting the ticks, since setting the ticks
        #actually adjusts the theta range window
        ax.set_thetamin(np.rad2deg(minTheta))
        ax.set_thetamax(np.rad2deg(maxTheta))

        ax.set_xlabel('redshift',size=14)
        ax.set_ylabel(thetaVal,size=14)

        ax.tick_params('both',labelsize=10.25)
        #ax.xaxis.grid(False)

        return ax


    def plotWedge(self,thetaVal,Zmin=0,Zmax=2,dTheta=2):

        if thetaVal == 'RA':
            center = self.fieldCenter[0]
        elif thetaVal == 'Dec':
            center = self.fieldCenter[1]
        else: center = None

        minTheta = np.deg2rad( center - dTheta )
        maxTheta = np.deg2rad( center + dTheta )

#        self.coneDiagram(thetaVal)
#        plt.title(self.fieldName)
        ax = self.coneDiagram(thetaVal, minZ=Zmin,maxZ=Zmax,rOrigin=-Zmin, minTheta=minTheta,maxTheta=maxTheta, overplot=True)
        plt.title(self.fieldName)

        return ax

    def wedgePatch(self,
        ax, r_min = 0.3, r_max = 0.5, t_min = np.pi/4, t_max = 3*np.pi/4
        ):
        ##see for explanation: https://stackoverflow.com/a/54400045

        ##some data
        #R = np.random.rand(100)*(r_max-r_min)+r_min
        #T = np.random.rand(100)*(t_max-t_min)+t_min
        #ax.scatter(T,R)

        ##compute the corner points of the wedge:
        axtmin = 0

        rs = np.array([r_min,  r_max,  r_min, r_max, r_min, r_max])
        ts = np.array([axtmin, axtmin, t_min, t_min, t_max, t_max])

        ##display them in a scatter plot --no
        #ax.scatter(ts, rs, color='r', marker='x', lw=5)

        ##from https://matplotlib.org/users/transforms_tutorial.html
        trans = ax.transData + ax.transAxes.inverted()

        ##convert to figure cordinates, for a starter
        xax, yax = trans.transform([(t,r) for t,r in zip(ts, rs)]).T

        '''
        for i,(x,y) in enumerate(zip(xax, yax)):
            ax.annotate(
                str(i), (x,y), xytext = (x+0.1, y), xycoords = 'axes fraction',
                arrowprops = dict(
                    width=2,

                ),
            )
        '''

        ##compute the angles of the wedge:
        tstart = np.rad2deg(angle(*np.array((xax[[0,1,2,3]],yax[[0,1,2,3]])).T))
        tend = np.rad2deg(angle(*np.array((xax[[0,1,4,5]],yax[[0,1,4,5]])).T))

        ##the center is where the two wedge sides cross (maybe outside the axes)
        center=seq_intersect(*np.array((xax[[2,3,4,5]],yax[[2,3,4,5]])).T)

        ##compute the inner and outer radii of the wedge:
        rinner = np.sqrt((xax[1]-center[0])**2+(yax[1]-center[1])**2)
        router = np.sqrt((xax[2]-center[0])**2+(yax[2]-center[1])**2)

        wedge = Wedge(center,
                      router, tstart, tend,
                      width=router-rinner,
                      #0.6,tstart,tend,0.3,
                      transform=ax.transAxes, linestyle='--', lw=3,
                      fill=False, color='red')
        ax.add_artist(wedge)


    def SFACTplots(self,thetaVal='RA'):

        ##full plot
        #ax = self.plotWedge(thetaVal,Zmin=0,Zmax=2)

        ##first window
        plot1radii = [0.04, 0.18]
        #Zmin1 = 0.04
        #Zmax1 = 0.18
        dTheta1 = 2      #degrees

        ax = self.plotWedge(thetaVal,plot1radii[0],plot1radii[1],dTheta1)

        #plot wedge patches around the appropriate redshifts
        center = self.getCenter(thetaVal)

        radiusDepth = [0.0521, 0.0658]    #Ha NB1
        angRange = np.deg2rad([center-0.5, center+0.5])

        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])

        radiusDepth = [0.1298, 0.1435]     #Ha NB3
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])


        ##second window
        Zmin2 = 0.28
        Zmax2 = 0.55
        dTheta2 = 1.5      #degrees

        ax = self.plotWedge(thetaVal,Zmin2,Zmax2,dTheta2)

        #wedge patches
        radiusDepth = [0.3072, 0.3571]    #OIII NB2
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])
        radiusDepth = [0.3791, 0.3970]    #OIII NB1
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])
        radiusDepth = [0.4809, 0.4989]    #OIII NB3
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])


        ##third window
        Zmin3 = 0.65
        Zmax3 = 1.05
        dTheta3 = 0.7      #degrees

        ax = self.plotWedge(thetaVal,Zmin3,Zmax3,dTheta3)

        #wedge patches
        radiusDepth = [0.7561, 0.7803]    #OII NB2
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])
        radiusDepth = [0.8527, 0.8768]    #OII NB1
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])
        radiusDepth = [0.9895, 1.0137]    #OII NB3
        self.wedgePatch(ax, radiusDepth[0],radiusDepth[1], angRange[0],angRange[1])


        ##fourth window
        Zmin4 = 1.3
        Zmax4 = 1.75
        dTheta4 = 0.5      #degrees

        ax = self.plotWedge(thetaVal,Zmin4,Zmax4,dTheta4)


    def fullRun(self):

        self.loadDatasets()
        self.cleanDatasets()
        self.loadCoords()

        self.SFACTplots('RA')
#        self.SFACTplots('Dec')

    def kNN(self, sourceIdx=1,targetIdx=2, zMin, zMax, RAmin, RAmax, Decmin, Decmax):
        '''
        ?
        Finds the nearest neighbor for sources within a dataset to targets within
        another dataset.
        The source dataset is limited to a central angular region (tMin & tMax)
        and a redshift region (zMin & zMax). The target dataset is unbounded
        (so neighbors may be found outside of the bounded source data region).

        The source & target datasets can be drawn from the same sample (i.e.,
        sourceIdx = targetIdx), but the above limiting criteria are still
        applied to the source dataset and not the target.

        Typically will be used to compute NN for a source & target, and then
        target & target to compare results.

        TODO: allow multiple datasets to be included in the target, so I can
        find an unbiased/unrestricted sample of neighbors.

        '''

        sourceData = np.array( self.dataCoords[sourceIdx] ).T
        targetData = np.array( self.dataCoords[targetIdx] ).T

        #trim the source & target data to consider only the desired sector:
        sourceDataTrim = sourceData[ (sourceData[:,2] > zMin) & (sourceData[:,2] < zMax) ] #redshift
        sourceDataTrim = sourceDataTrim[ (sourceDataTrim[:,0] > RAmin) & (sourceDataTrim[:,0] > RAmax) ] #RA
        sourceDataTrim = sourceDataTrim[ (sourceDataTrim[:,1] > Decmin) & (sourceDataTrim[:,1] > Decmax) ] #Dec






        numSource = len(sourceData)

        targetTree = cKDTree(targetData)

        NN_dists = np.zeros(numSource)
        NN_coords = np.zeros([numSource,3])

        if sourceIdx == targetIdx:
            #then, get the 2nd nearest neighbor since the 1st is itself
            kIdx = 2
        else:
            kIdx = 1

        for i in range(len(sourceData)):
            curCoord = sourceData[i]

            NN_dist, NN_idx = targetTree.query(curCoord,k=[kIdx])

            NN_coord = np.squeeze( targetData[NN_idx] )
            NN_dists[i] = self.NNdist(curCoord,NN_coord)
            NN_coords[i] = NN_coord

        return NN_dists,NN_coords

    def NNdist(self, coord1,coord2):


        RA1,Dec1,z1 = coord1
        RA2,Dec2,z2 = coord2


        #use AstroPy to get the angular distance between the points, and then
        #do the full distance computation myself due to Astropy's precision issues
        #with redshift & distances

        skycoord1 = SkyCoord(ra=RA1*u.rad, dec=Dec1*u.rad)
        skycoord2 = SkyCoord(ra=RA2*u.rad, dec=Dec2*u.rad)

        angDist = skycoord1.separation(skycoord2)


        d2 = (z1+z2)**2 * (np.sin(angDist/2.))**2 + (abs(z2-z1))**2 * (np.cos(angDist/2.))**2
        c_kmPerSec = c.to('km/s')
        H0 = Planck15.H0

        dist = np.sqrt(d2)*c_kmPerSec/H0   #Mpc

        return np.array(dist)

    def NNdist_astropy(self,coord1,coord2):
        '''
        I am dubious of astropy's precision with redshifts & distances, so this
        is just here for posterity but not for a use.
        '''

        RA1,Dec1,z1 = coord1
        RA2,Dec2,z2 = coord2

        dist1 = Distance(z = z1, cosmology = Planck15)
        dist2 = Distance(z = z2, cosmology = Planck15)

        skycoord1 = SkyCoord(ra=RA1*u.rad, dec=Dec1*u.rad, distance=dist1)
        skycoord2 = SkyCoord(ra=RA2*u.rad, dec=Dec2*u.rad, distance=dist2)

        #do a comparison between me and astropy, where only the z and Dec are different btwn points:
        #dist2 = skycoord1.separation_3d( SkyCoord(ra=RA1*u.rad,dec=Dec2*u.rad, distance=dist2)  )
        dist2 = skycoord1.separation_3d(skycoord2)   #distance btwn real points

        #then my method:
        alpha = Dec1 - Dec2
        d2 = (z1+z2)**2 * (np.sin(alpha/2.))**2 + (z2-z1)**2 * (np.cos(alpha/2.))**2

        c_kmPerSec = c.to('km/s')
        H0 = Planck15.H0

        dist1 = np.sqrt(d2)*c_kmPerSec/H0   #Mpc

        return dist1-dist2


datapath = '/home/bscousin/AstroResearch/thesis/data'
fname55 = 'hadot055_sfact.dat'
fname55_2 = 'HADot055_redshiftsSWB.txt'
fname55_SDSS = 'hadot055_SDSS_5deg.csv'

cone55 = coneData(datapath,[fname55_SDSS,fname55,fname55_2],'HaDot55',fieldCenter=[26.16670584,27.90981659])
#cone55.fullRun()
#cone55.SFACTplots()


fname22 = 'hadot022_sfact.dat'
fname22_2 = 'HADot022_redshiftsSWB.txt'
fname22_SDSS = 'hadot022_SDSS_5deg.csv'

cone22 = coneData(datapath,[fname22_SDSS,fname22,fname22_2],'HaDot22',fieldCenter=[39.80166399,	27.86709320])
cone22.fullRun()
#cone22.SFACTplots()


#test the NN stuff
sourceIdx=1; targetIdx=2
sourceData = np.array( cone22.dataCoords[sourceIdx] ).T
targetData = np.array( cone22.dataCoords[targetIdx] ).T


NN_dists22, NN_coords22 = cone22.kNN()
#NN_dists22_self, NN_coords22_self = cone22.kNN(sourceIdx=2,targetIdx=2)




coord1 = sourceData[-1]
coord2 = NN_coords22[-1]
