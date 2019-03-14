'''
Backup from before I made the (clunky) changes to accommodate Sam's separate dataset
(changes included a lot of clunky array & list things...)
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:46:04 2019

@author: bscousin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Wedge

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, IllegalHourError

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


class coneData():


    def __init__(self, datapath, fnameSDSS=None, fnameSURVEY=None, fieldName='', fieldCenter=[26.16670584,27.90981659], thetaVal='RA'):
        #note: I'm assuming through this class that /datapath contains both the
        #SDSS data and the other data.
        self.datapath = datapath
        self.fnameSDSS = fnameSDSS
        self.fnameSURVEY = fnameSURVEY
        self.fieldName = fieldName
        self.fieldCenter = fieldCenter
        self.thetaVal = thetaVal
        if thetaVal == 'RA':
            self.center = fieldCenter[0]
        elif thetaVal == 'DEC':
            self.center = fieldCenter[1]
        else:
            self.center = None

        self.SDSScoords = None
        self.SURVEYcoords = None
        #?


    def setSDSSfile(self, fnameSDSS):
        self.fnameSDSS = fnameSDSS

    def setSURVEYfile(self, fnameSURVEY):
        self.fnameSURVEY = fnameSURVEY

    def loadSDSSdata(self, headerLine=1):
        '''
        Load general galaxy position data from a .csv formatted by SkyServer into a
        Pandas dataframe.
        '''

        #if splitext(self.datapath)[1] == '.csv':
        try:
            datapathSDSS = join(self.datapath, self.fnameSDSS)
            SDSSdata = pd.read_csv(datapathSDSS, header=headerLine)
        except TypeError:
            print('You must first specify the data file using coneData.setSDSSfile()')
            return
        self.SDSSdata = SDSSdata


    def fieldParser(data, fields):
        for x in fields:
            try:
                vals = np.array(data[x])
                field = x
                return vals,field
            except KeyError:
                vals = np.array([])
                field = ''
                continue
        return vals, field


    def cleanSDSSdata(self):
        pass


    def loadSURVEYdata(self,headerLine=1):
        '''
        Load general galaxy position data from a .dat formatted by the SFACT crew
        into a Pandas dataframe.
        '''

        try:
            datapathSURVEY = join(self.datapath, self.fnameSURVEY)
            SURVEYdata= pd.read_csv(datapathSURVEY, sep='\s+', header=headerLine)
        except TypeError:
            print('You must first specify the data file using coneData.setSURVEYfile()')
            return
        self.SURVEYdata = SURVEYdata



    def cleanSURVEYdata(self):
        '''
        Filters/masks galaxy redshift/coordinate data based on certain flags.
        '''

        nullVal = -9.99990
        nullStr = 'INDEF'
        nullType = 17

        #RAfields = ['ra','RA_DEG','RA']
        #DECfields = ['dec', 'DEC_DEG','DEC']
        #Zfields = ['z','redshift','Z_EST']

        #RAs, RAfield = self.fieldParser(self.SURVEYdata,RAfields)
        #DECs, DECfield = self.fieldParser(self.SURVEYdata,DECfields)
        #Zs, Zfield = fieldParser(self.SURVEYdata,Zfields)

        RAfield = 'RA'
        DECfield = 'DEC'
        Zfield = 'Z_EST'

        #RAs = self.SURVEYdata[RAfield]
        #DECs = self.SURVEYdata[DECfield]
        #Zs = self.SURVEYdata[Zfield]

        #Zs = np.array(data[Zfield])
        typefield = 'ELGTYPE'

        SURVEYdata = self.SURVEYdata

        #ensure that the data is in floats, not strings of numbers......
        SURVEYdata[Zfield] = pd.to_numeric(SURVEYdata[Zfield], errors='coerce')
        #data[typefield] = data[typefield].astype('float')
        SURVEYdata[typefield] = pd.to_numeric(SURVEYdata[typefield], errors='coerce', downcast='integer')
        #only 'INDEF' fields will be coerced into NaNs, so drop them:
        SURVEYdata = SURVEYdata.dropna(subset = [typefield])

        #then, drop rows that have null values for redshift or are marked as a wrong detection
        goodSURVEYdata = SURVEYdata[ (SURVEYdata[Zfield] != nullVal) & (SURVEYdata[typefield] != nullType)]

        self.goodSURVEYdata = goodSURVEYdata

        #goodRAs = goodData[RAfield]
        #goodDECs = goodData[DECfield]
        #goodZs = goodData[Zfield]
        #return goodRAs,goodDECs,goodZs


    def angleConv(self,thetaVals,dataFormat):

        if dataFormat=='SDSS':
            #SDSS has RA/DEC in degrees
            thetaDeg = Angle(thetaVals, u.degree)
            thetaRad = thetaDeg.radian
        elif dataFormat=='SFACT':
            #SFACT has RA in hours, DEC in degrees
            #try RA hour conversion first; otherwise, it's DEC in degrees
            try:
                thetaHours = Angle(thetaVals, u.hour)
                thetaRad = thetaHours.radian
            except IllegalHourError:
                thetaDegs = Angle(thetaVals, u.degree)
                thetaRad = thetaDegs.radian
        return thetaRad

    def loadCoords(self,dataFormat):
        '''
        ??

        Careful... this requires that the data (SFACT particularly) is cleaned
        so that any odd values (like strings for RA) are removed.
        '''
        if dataFormat=='SDSS':
            RAfield = 'ra'
            DECfield = 'dec'
            Zfield = 'redshift'

            RAs = self.SDSSdata[RAfield]
            DECs = self.SDSSdata[DECfield]
            Zs = self.SDSSdata[Zfield]

            RAs = self.angleConv(RAs,dataFormat)
            DECs = self.angleConv(DECs,dataFormat)

            self.SDSScoords = (RAs,DECs,Zs)

        elif dataFormat=='SFACT':
            RAfield = 'RA'
            DECfield = 'DEC'
            Zfield = 'Z_EST'

            RAs = self.goodSURVEYdata[RAfield]
            DECs = self.goodSURVEYdata[DECfield]
            Zs = self.goodSURVEYdata[Zfield]

            RAs = self.angleConv(RAs,dataFormat)
            DECs = self.angleConv(DECs,dataFormat)

            self.SURVEYcoords = (RAs,DECs,Zs)



        #RAfields = ['ra','RA_DEG','RA']
        #DECfields = ['dec', 'DEC_DEG','DEC']
        #Zfields = ['z','redshift','Z_EST']

        #RAs, RAfield = self.fieldParser(self.SURVEYdata,RAfields)
        #DECs, DECfield = self.fieldParser(self.SURVEYdata,DECfields)
        #Zs, Zfield = fieldParser(self.SURVEYdata,Zfields)

        return RAs,DECs,Zs



    def coneDiagram(self,dataFormat, minTheta=None, maxTheta=None, minZ=0, maxZ=2, rOrigin=0, overplot=False):

        if dataFormat=='SDSS':
            RAs,DECs,Zs = self.SDSScoords
        elif dataFormat=='SFACT':
            RAs,DECs,Zs = self.SURVEYcoords
        else:
            pass

        if self.thetaVal=='RA':
            thetaRad = RAs
        elif self.thetaVal=='DEC':
            thetaRad = DECs
        else:
            raise ValueError('Specify a valid coordinate for theta (RA or DEC)')

        if overplot:
            fig = plt.gcf()
            color = 'g'
            m = '*'
            #size = 20*2.5
            size = 8
        else:
            fig = plt.figure()
            color = 'k'
            m = 'o'
            #size = 1*6
            size = 1


        ax = fig.add_subplot(111, projection='polar')
        plt.scatter(thetaRad,Zs, c=color, marker=m, s = size)

        ax.set_rmax(maxZ)
        ax.set_rmin(minZ)
        ax.set_rorigin(rOrigin)

        #set RA limits to be scaled from the dataset's own limits
        if minTheta is None:
            minTheta = 0.9*min(thetaRad)

        if maxTheta is None:
            maxTheta = 1.1*max(thetaRad)
            if maxTheta > 2*np.pi: maxTheta = 2*np.pi

        ax.set_thetamin(np.rad2deg(minTheta))
        ax.set_thetamax(np.rad2deg(maxTheta))

        ticks = np.linspace(minTheta, maxTheta, 4)
        ax.set_xticks(ticks)

        ax.set_xlabel('redshift',size=14)
        ax.set_ylabel('right ascension',size=14)
        ax.tick_params('both',labelsize=10.25)
        ax.xaxis.grid(False)



    def plotWedge(self,Zmin,Zmax,dTheta):

        minTheta = np.deg2rad( self.center - dTheta )
        maxTheta = np.deg2rad( self.center + dTheta )

        self.coneDiagram('SDSS')
        plt.title(self.fieldName)
        #self.coneDiagram('SFACT', minZ=Zmin,maxZ=Zmax,rOrigin=-Zmin, minTheta=minTheta,maxTheta=maxTheta, overplot=True)
        #plt.title(self.fieldName)

    def wedgePatch(self, radiusDepth, plotRadius, dTheta=0.5):

        ##Add a wedge

        #define the wedge's depth, angular range, and width

        ax = plt.gca()

        angRange = [self.center-dTheta, self.center+dTheta]
        #angRange = np.rad2deg(angRange)

        wedgeCenter = (0.5,0.5)
        r = radiusDepth[1]
        width = radiusDepth[1]-radiusDepth[0]

        r_inner = transformRadius(r-width, plotRadius[0], plotRadius[1])
        r_outer = transformRadius(r, plotRadius[0], plotRadius[1])

        wedge = Wedge(
            wedgeCenter,
            r_outer,
            angRange[0],angRange[1],
            width=r_outer-r_inner,
            transform=ax.transAxes, linestyle='--',
            fill=False, color='red'
        )
        '''Testing Wedge parameters........
        wedge = Wedge(
            wedgeCenter,
            0.5,
            angRange[0],angRange[1],
            #width=r_outer-r_inner,
            transform=ax.transAxes, linestyle='--',
            fill=False, color='red'
        )
        '''
        ax.add_artist(wedge)


    def SFACTplots(self):

        #first window
        plot1radii = [0.04, 0.18]
        #Zmin1 = 0.04
        #Zmax1 = 0.18
        dTheta1 = 2      #degrees

        self.plotWedge(plot1radii[0],plot1radii[1],dTheta1)

        #radiusDepth = [0.1298, 0.1435]
        #self.wedgePatch(radiusDepth, plot1radii)


        '''
        #second window
        Zmin2 = 0.28
        Zmax2 = 0.55
        dTheta2 = 1.5      #degrees

        self.plotWedge(Zmin2,Zmax2,dTheta2)

        #third window
        Zmin3 = 0.65
        Zmax3 = 1.05
        dTheta3 = 0.7      #degrees

        self.plotWedge(Zmin3,Zmax3,dTheta3)

        #fourth window
        Zmin4 = 1.3
        Zmax4 = 1.75
        dTheta4 = 0.5      #degrees

        self.plotWedge(Zmin4,Zmax4,dTheta4)
        '''

    def fullRun(self):

        #cone = ConeData(datapath,fname1,fname2)
        self.loadSDSSdata()
        self.loadSURVEYdata()
        self.cleanSURVEYdata()

        #RAs_SDSS, DECs_SDSS, Zs_SDSS = self.loadCoords('SDSS')
        #RAs_SURVEY, DECs_SURVEY, Zs_SURVEY = self.loadCoords('SFACT')

        #self.coneDiagram(RAs_SDSS,DECs_SDSS,Zs_SDSS, minTheta=None, maxTheta=None, minZ=0, maxZ=2, rOrigin=0, thetaVal='RA', overplot=False)



datapath = '/home/bscousin/AstroResearch/thesis/data'
fname55 = 'hadot055_sfact.dat'
fname55_SDSS = 'hadot055_SDSS_5deg.csv'

cone55 = coneData(datapath,fname55_SDSS,fname55,'HaDot55',fieldCenter=[26.16670584,27.90981659])

cone55.loadSDSSdata()
cone55.loadSURVEYdata()
cone55.cleanSURVEYdata()

cone55.loadCoords('SDSS')
RAs_SDSS, DECs_SDSS, Zs_SDSS = cone55.SDSScoords

cone55.loadCoords('SFACT')
RAs_SURVEY, DECs_SURVEY, Zs_SURVEY = cone55.SURVEYcoords


cone55.SFACTplots()

#cone.coneDiagram('SDSS', minTheta=None, maxTheta=None, minZ=0, maxZ=2, rOrigin=0, thetaVal='RA', overplot=False)
#cone.coneDiagram('SFACT', minTheta=None, maxTheta=None, minZ=0, maxZ=2, rOrigin=0, thetaVal='RA', overplot=True)




#testCone.fullRun()
#testCone.loadSURVEYdata()
