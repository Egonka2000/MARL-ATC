from numpy import arctan2,degrees,array,sqrt # to allow arrays, their functions and types
from windfield import *

class WindSim(Windfield):
    def add(self, *arg):

        lat = arg[0]
        lon = arg[1]
        winddata = arg[2:]

        ndata = len(winddata)

        # No altitude or just one: same wind for all altitudes at this position
        
        if ndata ==3 or (ndata==4 and winddata[0]==None): # only one point, ignore altitude 
            self.addpoint(lat,lon,winddata[1],winddata[2])

        # More than one altitude is given
        elif ndata>3:
            windarr = array(winddata)
            dirarr = windarr[1::3]
            spdarr = windarr[2::3]
            altarr = windarr[0::3]
            
            self.addpoint(lat,lon,dirarr,spdarr,altarr)

        elif winddata.count("DEL")>0:
            self.clear()
            
        else:# Something is wrong
            return False,"Winddata not recognized"
        
        return True
    
    def get(self,lat,lon,alt=None):

        vn,ve = self.getdata(lat,lon,alt)

        wdir = degrees(arctan2(ve,vn))%360.
        wspd = sqrt(vn*vn+ve*ve)
        
        txt  = "WIND AT %.5f, %.5f: %03d/%d" % (lat,lon,wdir,wspd)

        return True,txt