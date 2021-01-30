def unPackRawFile(raw_path):
    """
    - unpacks the Raw conditions file. Not used for the neural network.
    """
    y = np.loadtxt(raw_path, skiprows=6)
    distance = y[:,0]
    pec_vel = y[:,1]
    temp = y[:,2]
    HI_density = y[:,3]
    #gas_density = y[:,4]
    #gas_metallicity = y[:,5]
    
    return distance, pec_vel, temp, HI_density#, gas_density, gas_metallicity

def unPackRawFlux(flux_path):
    """
    - unpacks the Raw conditions file. Not used for the neural network.
    """
    y2 = np.genfromtxt(flux_path, skip_header=1, delimiter=' , ')
    velocity = y2[:,0]
    flux = y2[:,1] 
    return velocity, flux

def hubble_flow_convert(velocity, a, omega_m, omega_lam):
    """
    - uses hubble flow to convert from velocity to distance
    """
    aH = a * 100 * (omega_m / a ** 3 + omega_lam)** 0.5
    return velocity/aH

def resample(distance, item, new_distance):
    """
    - interpolates the distances so that we can resample. useful because the velocity after converting using hubble flow doesn't have the same positions as the underlying properties.
    - creates a consistent distance scale (obviously these distances are messed up by peculiar velocities)
    """
    f = interp.interp1d(distance, item)
    new_item = f(new_distance)
    
    return new_item

def Box(lx,ly,lz, Lbox, h100, aexp, ms, xg, yg, zg, pos, e3):
    """
    Checks the periodic image of 1 box, helper function for get galaxies.
    """
    # galaxy positions in this peridodic image
    x = xg + Lbox*lx
    y = yg + Lbox*ly
    z = zg + Lbox*lz
    
    # projections of galaxy position on LOS
    w = e3[0]*(x-pos[0]) + e3[1]*(y-pos[1]) + e3[2]*(z-pos[2])
    dx = x - pos[0] - w*e3[0]
    dy = y - pos[1] - w*e3[1]
    dz = z - pos[2] - w*e3[2]
    
    # distance from the galaxy to the LOS
    dr = np.sqrt(dx**2+dy**2+dz**2)
    
    # convert dr from comoving Mpc/h to proper Mpc
    dr *= (aexp/h100)
    #print((aexp/h100))
    # select galaxies
    sel1 = np.logical_and(w>2.19,w<97.7) # within first 120 cMpc/h
    sel2 = np.logical_and(dr<(3),ms>1.0e7) # within 4 pMpc of the LOS and mass > 1e9 Msun
    sel = np.logical_and(sel1,sel2)
    
    ms_return = np.array([])
    w1_return = np.array([])
    dr_return = np.array([])
    
    if(len(sel) > 0):
        w1 = np.compress(sel,w)
        ms1 = np.compress(sel,ms)
        dr1 = np.compress(sel,dr)
        #for i in range(len(ms1)):
            #print('ms=%.3e l=%5.1f dr=%.2f'%(ms1[i],w1[i],dr1[i]))

        w1_return = np.append(w1, w1)
        ms_return = np.append(ms1, ms1)
        dr_return = np.append(dr1, dr1)
        #gives mass, distance along LOS, and distance away from LOS
    return w1_return, ms_return, dr_return
            
         ##
    ##
##

# check 3x3x3 periodic images around the main box to account for periodic BC

def getDir(path):
    """
    the direction of the LOS is given inside each file, (in the comments)
    this function parses the comments to get that information
    """
    length = len(list(open(path)))
    x = np.genfromtxt(path, comments='nothing', skip_header=5, skip_footer=(length-6), delimiter='', dtype=str)

    answer = re.search('\(([^)]+)', x[2]).group(1)

    arr = list(answer.split(','))

    return float(arr[0]), float(arr[1])

def getPos(path):
    """
    the start position of the LOS is given inside each file, (in the comments)
    this function parses the comments to get that information
    """
    length = len(list(open(path)))
    x = np.genfromtxt(path, comments='nothing', skip_header=5, skip_footer=(length-6), delimiter='', dtype=str)

    answer = re.search('\(([^)]+)', x[1]).group(1)

    arr = list(answer.split(','))
    
    return float(arr[0]), float(arr[1]), float(arr[2])

def convertSphereToCart(theta, phi):
    "converts a unit vector in spherical to cartesian, needed for getGalaxies"
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)


def get_galaxies(path_galaxies, path_LOS, number):
    """
    function can take in the path of the galaxy file (.res) and the line of sight. Number should match the LOS #
    """
    ms_full = np.array([])
    w1_full = np.array([])
    dr_full = np.array([])
    num_full = np.array([])
    
    # Parameter
    Lbox = 40  # box size
    aexp = 0.1452  # scale factor for the epoch
    h100 = 0.6814    # normalized Hubble constant
    pos = np.array(getPos(path_LOS))  # origin of LOS
    pos = pos/256*40
    sphere_los = np.array(getDir(path_LOS))  # direction of LOS , np.sum(e3**2) should be = 1
    e3 = convertSphereToCart(sphere_los[0], sphere_los[1])
    x = np.loadtxt(path_galaxies, usecols=[1,5,6,7],unpack=1) # load data

    ms = x[0] # stellar mass
    xg = x[1] #\
    yg = x[2] # | positions of the galaxies in cMpc/h
    zg = x[3] #/
    
    for lz in range(-1,3):
        for ly in range(-1,3):
            for lx in range(-1,3):
                w1_temp, ms_temp, dr_temp = Box(lx,ly,lz, Lbox, h100, aexp, ms, xg, yg, zg, pos, e3)
                dr_temp = dr_temp#/(aexp/h100)
                w1_full = np.append(w1_full, w1_temp)
                ms_full = np.append(ms_full, ms_temp)
                dr_full = np.append(dr_full, dr_temp)
                num_full = np.append(num_full, number*np.ones(len(w1_temp)))

    
    return np.array([ms_full, w1_full, dr_full, num_full])

