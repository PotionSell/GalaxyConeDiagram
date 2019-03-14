#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:15:32 2019

@author: bscousin
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

savepath = '/home/bscousin/OneDrive/MiscCollege/AstroResearch/thesis/figures'

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


class sectorClass():

    def __init__(self,radii=[None,None],thetas=[None,None]):
        '''
        Create a sector for easy referencing (e.g., creating a wedge sector
        plot patch, selecting a region of data, etc.)
        '''
        self.rMin = radii[0]
        self.rMax = radii[1]
        self.tMin = thetas[0]
        self.tMax = thetas[1]

    def getBounds(self):
        return self.rMin,self.rMax,self.tMin,self.tMax


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

            self.dataCoords[i] = np.array((RAs,DECs,Zs))


    def coneDiagram(self, thetaVal='RA', sector=sectorClass(), rOrigin=0, overplot=False):

        colors = ['k','g','b','r']
        markers = ['o','*','*','*']
        sizes = [2,8,8,8]

        fig = plt.figure(figsize=(12,13))
        ax = fig.add_subplot(111, projection='polar')

        for i in range(self.numDatasets):

            (RAs,DECs,Zs) = self.dataCoords[i]

            if thetaVal=='RA':
                thetaRad = RAs
            elif thetaVal=='Dec':
                thetaRad = DECs
            else:
                raise ValueError('Specify a valid coordinate for theta (RA or DEC)')

            ax.scatter(thetaRad,Zs, c=colors[i], marker=markers[i], s = sizes[i],edgecolors='k',linewidths=0.1)
            #note: this is NOT plotting each point individually; this plots
            #each dataset individually

        zMin,zMax,tMin,tMax = sector.getBounds()

        #ax.set_rmax(maxZ)
        #ax.set_rmin(minZ)
        ax.set_ylim([zMin,zMax])
        ax.set_rorigin(rOrigin)

        #set theta limits to be scaled from the dataset's own limits
        #note: this will be from the SDSS dataset since it has the largest range
        if tMin is None:
            tMin = 0.9*min(thetaRad)

        if tMax is None:
            tMax = 1.1*max(thetaRad)
            if tMax > 2*np.pi: tMax = 2*np.pi

        ticks = np.linspace(tMin, tMax, 4)
        ax.set_xticks(ticks)

        #convert to decimal precision
        ax.xaxis.set_major_formatter(FuncFormatter(rad2fmt))

        #you must call these AFTER setting the ticks, since setting the ticks
        #actually adjusts the theta range window
        ax.set_thetamin(np.rad2deg(tMin))
        ax.set_thetamax(np.rad2deg(tMax))

        ax.set_xlabel('redshift', rotation=tMin,size=14)

        rPos = transformRadius(zMax,zMin,zMax)

        ax.set_ylabel(thetaVal, rotation=tMin-90,size=14)

        ax.tick_params('both',labelsize=10.25)
        #ax.xaxis.grid(False)

        return ax


    def plotWedge(self,thetaVal,radii=[0,2],dTheta=2):

        if thetaVal == 'RA':
            center = self.fieldCenter[0]
        elif thetaVal == 'Dec':
            center = self.fieldCenter[1]
        else: center = None

        minTheta = np.deg2rad( center - dTheta )
        maxTheta = np.deg2rad( center + dTheta )

        sector = sectorClass(radii,[minTheta,maxTheta])
#        self.coneDiagram(thetaVal)
#        plt.title(self.fieldName)
        ax = self.coneDiagram(thetaVal,sector,rOrigin=-radii[0], overplot=True)
        plt.title(self.fieldName)

        return ax

    def wedgePatch(self, ax, sector,wedgeLabel=''):
        ##see for explanation: https://stackoverflow.com/a/54400045

        ##compute the corner points of the wedge:
        axtmin = 0

        r_min,r_max,t_min,t_max = sector.getBounds()

        rs = np.array([r_min,  r_max,  r_min, r_max, r_min, r_max])
        ts = np.array([axtmin, axtmin, t_min, t_min, t_max, t_max])

        ##display them in a scatter plot --no
        #ax.scatter(ts, rs, color='r', marker='x', lw=5)

        ##from https://matplotlib.org/users/transforms_tutorial.html
        trans = ax.transData + ax.transAxes.inverted()

        ##convert to figure cordinates, for a starter
        xax, yax = trans.transform([(t,r) for t,r in zip(ts, rs)]).T


        #adapt this to label the different wedges with emission lines???
        #BSC: label the wedge bounds
        for i,(x,y) in enumerate(zip(xax, yax)):
            #only want to label the third bound point:
            if i%3 == 0:
                plt.annotate(wedgeLabel, (x,y),xytext = (x+0.01,y),
                             xycoords='axes fraction',
                             bbox=dict(facecolor='white',edgecolor='black')
                             )

                #or do an arrow...
                '''
                ax.annotate( wedgeLabel, (x,y), xytext = (x+0.1,y),
                            xycoords='axes fraction',
                            arrowprops=dict(width=2,color='lightgray')
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
                      fill=False, color='gray')
        ax.add_artist(wedge)


    def SFACTplots(self,
                   thetaVal='RA',
                   plotRadii=[[0.04,0.18],[0.28,0.55],[0.65,1.05]],
                   sectorRadii=[[0.0521, 0.0658],[0.1298, 0.1435],
                                [0.3072, 0.3251],[0.3791, 0.3970],
                                [0.4809, 0.4989],[0.7561, 0.7803],
                                [0.8527, 0.8768],[0.9895, 1.0137]
                                ],
                   sectorLabels=['Ha NB1','Ha NB3',
                                 'OIII NB2', 'OIII NB1',
                                 'OIII NB3', 'OII NB2',
                                 'OII NB1', 'OII NB3'
                                 ]
                   ):
        '''
        Default redshift windows are:
            [0.0521, 0.0658]    #Ha NB1
            [0.1298, 0.1435]    #Ha NB3
            [0.3072, 0.3251]    #OIII NB2
            [0.3791, 0.3970]    #OIII NB1
            [0.4809, 0.4989]    #OIII NB3
            [0.7561, 0.7803]    #OII NB2
            [0.8527, 0.8768]    #OII NB1
            [0.9895, 1.0137]    #OII NB3
        Maybe do something about the whole dTheta angular size of the cone diagram...
        I originally adjusted it for each window:
            [0.04,0.18]    2.0 deg
            [0.28,0.55]    1.5 deg
            [0.65,1.05]    0.7 deg
            [1.30,1.75]    0.5 deg
        but I now set it to be 2 degrees for all of them. Could this be computed dynamically?

        '''

        center = self.getCenter(thetaVal)
        angRange = np.deg2rad([center-0.5, center+0.5])

        for i in range(len(plotRadii)):
            plotRads = plotRadii[i]

            ax = self.plotWedge(thetaVal,plotRads,2)

            for j in range(len(sectorRadii)):
                sectorRads = sectorRadii[j]
                #draw only the wedges within the plot, so check the radii bounds
                if (sectorRads[1] < plotRads[1] and sectorRads[0] > plotRads[0]):
                    sector = sectorClass(sectorRads,angRange)
                    self.wedgePatch(ax,sector,sectorLabels[j])
            figName = self.fieldName + '_' + thetaVal + 'wedge' + str(i+1)

            rez = 400
            plt.savefig(join(savepath, figName),dpi=rez)


    def kNN(self, sectorRA, sectorDec ,sourceIdx=[1],targetIdx=[0,2]):
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

        '''

        zMin, zMax, RAmin, RAmax = sectorRA.getBounds()
        _,_,Decmin,Decmax = sectorDec.getBounds()

        #start off with the first dataset, then concatenate more if they're specified:
        i = 0; j = 0
        for x in sourceIdx:
            if i == 0:
                sourceData = self.dataCoords[x]
            else:
                sourceData = np.concatenate((sourceData,self.dataCoords[x]),axis=1)
            i += 1
        for x in targetIdx:
            if j == 0:
                targetData = self.dataCoords[x]
            else:
                targetData = np.concatenate((targetData,self.dataCoords[x]),axis=1)
            j += 1

        sourceData = sourceData.T
        targetData = targetData.T

        #trim the source & target data to consider only the desired sector:
        idx1 = (sourceData[:,2] > zMin) & (sourceData[:,2] < zMax)        #redshift
        idx2 = (sourceData[:,0] > RAmin) & (sourceData[:,0] < RAmax)      #RA
        idx3 = (sourceData[:,1] > Decmin) & (sourceData[:,1] < Decmax)    #Dec

        sourceDataTrim = sourceData[idx1 & idx2 & idx3]

        numSource = len(sourceDataTrim)

        targetTree = cKDTree(targetData)

        NN_dists = np.zeros(numSource)
        NN_coords = np.zeros([numSource,3])

        kIdx = 1

        for i in range(numSource):
            curCoord = sourceDataTrim[i]

            NN_dist, NN_idx = targetTree.query(curCoord,k=[kIdx])

            #the distance will be zero if a point is in both source & target sets
            if NN_dist == 0:
                #so, get the 2nd NN since the 1st NN is itself
                NN_dist, NN_idx = targetTree.query(curCoord,k=[kIdx+1])

            NN_coord = np.squeeze( targetData[NN_idx] )
            NN_dists[i] = self.skyDist(curCoord,NN_coord)
            NN_coords[i] = NN_coord

        return NN_dists,NN_coords

    def skyDist(self, coord1,coord2):

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

    def runNN(self,
               radii=[[0.0521, 0.0658],[0.1298, 0.1435],[0.3072, 0.3251],[0.3791, 0.3970],[0.4809, 0.4989],[0.8527, 0.8768]],
               sourceIdx=[1],targetIdx=[0,2]
               ):
        '''

        '''
        centerRA = self.getCenter('RA')
        centerDec = self.getCenter('Dec')

        angRangeRA = np.deg2rad([centerRA-0.5, centerRA+0.5])
        angRangeDec = np.deg2rad([centerDec-0.5, centerDec+0.5])

        radii = np.array(radii)

        for i in range(len(radii)):
            x = radii[i]
            sectorRA = sectorClass(x,angRangeRA)
            sectorDec = sectorClass(x,angRangeDec)

            NN_dists,NN_coords = self.kNN(sectorRA,sectorDec,sourceIdx=sourceIdx,targetIdx=targetIdx)
            #NN_dists,NN_coords = self.kNN(sectorRA,sectorDec,sourceIdx=[1],targetIdx=[0,1,2])

            #set up histogram
            plt.figure()

            dMax = 100 #Mpc
            binWidth = 1
            bins = np.arange(0,dMax,binWidth)

            plt.hist(NN_dists,bins=bins)

            #scale x-axis to be smaller relevant distances, but still include
            #plots that go beyond this range
            dMin = 26   #Mpc
            if (len(NN_dists) != 0) and (np.min(NN_dists) <= dMin):
                plt.xlim([0,dMin])

            plt.xlabel('distance (Mpc)')
            plt.ylabel('number')

            if sourceIdx==targetIdx:
                title = 'NN for Background-Background Galaxies between redshifts: ' +str(x)
                figName = self.fieldName + '_BG_NN' + str(i+1)
            else:
                title = 'NN for ELG-Background Galaxies between redshifts: ' +str(x)
                figName = self.fieldName + '_ELG_NN' + str(i+1)

            plt.title(title)
            plt.savefig(join(savepath, figName),dpi=200)

        return NN_dists,NN_coords


    def fullRun(self):

        self.loadDatasets()
        self.cleanDatasets()
        self.loadCoords()

        self.SFACTplots('RA')
#        self.SFACTplots('Dec')


#datapath = '/home/bscousin/AstroResearch/thesis/data'
datapath = '~/OneDrive/MiscCollege/AstroResearch/thesis/data'
fname55 = 'hadot055_sfact.dat'
#fname55_2 = 'HADot055_redshiftsSWB.txt'
fname55_2 = 'HADot055_redshiftsSWB_030919.txt'
fname55_SDSS = 'hadot055_SDSS_5deg.csv'

cone55 = coneData(datapath,[fname55_SDSS,fname55,fname55_2],'HaDot55',fieldCenter=[26.16670584,27.90981659])
cone55.fullRun()
#NN_ELGdists55,NN_ELGcoords55 = cone55.runNN(sourceIdx=[1],targetIdx=[0,2])
#NN_BGdists55,NN_BGcoords55 = cone55.runNN(sourceIdx=[0,2],targetIdx=[0,2])

#NN_ELGdists55,NN_ELGcoords55 = cone55.runNN(radii=[[0.3791,0.397]],sourceIdx=[1],targetIdx=[0,2])

fname22 = 'hadot022_sfact.dat'
fname22_2 = 'HADot022_redshiftsSWB_030919.txt'
fname22_SDSS = 'hadot022_SDSS_5deg.csv'

cone22 = coneData(datapath,[fname22_SDSS,fname22,fname22_2],'HaDot22',fieldCenter=[39.80166399,	27.86709320])
#cone22.fullRun()

#NN_ELGdists22,NN_ELGcoords22 = cone22.runNN(sourceIdx=[1],targetIdx=[0,2])
#NN_BGdists22,NN_BGcoords22 = cone22.runNN(sourceIdx=[0,2],targetIdx=[0,2])

