#!/usr/local/bin/python3
#
# Programme aimed on measurements of zeeman shift in stellar polarized spectra
# by means of the center of gravity method and gaussian fitting of line's profile
# Author: Eugene Semenko. Last modification: 15 Jan 2021

from sys import exit, argv
from numpy import arange, where, zeros, exp, mean, std, savetxt, vstack, sqrt, log2, array, absolute
from astropy.io.fits import getheader, getdata
from scipy import interpolate, integrate
from scipy.optimize import curve_fit, leastsq
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Button, Slider, Cursor

glande = 1.23

# General functions
def interpol_spec(w, r, wl, d):
	tck = interpolate.splrep(w, r, s=0)
	ri = interpolate.splev(wl, tck, der=d)
	return ri

def contin_def(w, r1):
	meanx = mean(w)
	leftarr = r1[where((w >= w[0]) & (w <= meanx))]
	y1 = max(leftarr)
	x1 = w[where(r1 == y1)]
	rightarr = r1[where((w >= meanx) & (w <= w[-1]))]
	y2 = max(rightarr)
	x2 = w[where(r1 == y2)]
	xnew = w[where((w > x1) & (w < x2))]
	ynew = r1[where((w > x1) & (w < x2))]
	k = (y2 - y1) / (x2 - x1)
	b = y1 - k * x1
#	print("(%.4f, %.2f) - (%.4f, %.2f) k = %.3f, b = %.3f" %(x1, y1, x2, y2, k, b))
	return k * xnew + b, xnew, ynew

gauss = lambda p, x: p[0] + p[1] * x - p[2] * exp(-(4.*log2(2) * (x - p[3])**2)/ p[4]**2)

errfunc = lambda p, x, y: y - gauss(p, x)

# Define classes
class ZSpec(object):
	def __init__(self):
		if len(argv) < 2:
			exit("Usage: %s <input polarized FITS file>" %argv[0])
		# Parse input filenames
		f1name = argv[1]
		if f1name.find('_1.f') != -1:
			f2name = f1name.replace('_1.', '_2.')
		elif f1name.find('_2.f') != -1:
			f2name = f1name.replace('_2', '_1')
		# Read spectra and fill up generalized data
		hdr1 = getheader(f1name); hdr2 = getheader(f2name)
		npix1 = hdr1['NAXIS1']; npix2 = hdr2['NAXIS1']
		crval1 = hdr1['CRVAL1']; crval2 = hdr2['CRVAL1']
		crpix1 = hdr1['CRPIX1']; crpix2 = hdr2['CRPIX1']
		cdelt1 = hdr1['CDELT1']; cdelt2 = hdr2['CDELT1']
		wl1 = crval1 + (arange(npix1) - crpix1) * cdelt1
		wl2 = crval2 + (arange(npix2) - crpix2) * cdelt2
		r1 = getdata(f1name); r2 = getdata(f2name)
		# Below are the class attributes
		self.wli = arange(max(wl1[0],wl2[0]), min(wl1[-1],wl2[-1]), cdelt1)
		self.ri1 = interpol_spec(wl1, r1, self.wli, 0)
		self.ri2 = interpol_spec(wl2, r2, self.wli, 0)
		self.istockes = (self.ri1 + self.ri2) / 2.0
		self.vstockes = (self.ri1 - self.ri2) / (self.ri1 + self.ri2)

class ZLine(ZSpec):
	def __init__(self):
		self.wl = None
		self.r1 = None
		self.r2 = None

	def change_range(self, wl0, width):
		self.wl = ZSpec().wli[where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		self.r1 = ZSpec().ri1[where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		self.r2 = ZSpec().ri2[where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
#		print("Line border changed to %.4f and %.4f" %(self.wl[0], self.wl[-1]))

	def measure_line(self):
		# Measure shift by COG method
		# Redetermination of continuum level
		self.cont1, self.nwl1, nr1 = contin_def(self.wl, self.r1)
		self.cont2, self.nwl2, nr2 = contin_def(self.wl, self.r2)
		y11 = self.nwl1 * (self.cont1 - nr1)
		y12 = (self.cont1 - nr1)
		y21 = self.nwl2 * (self.cont2 - nr2)
		y22 = (self.cont2 - nr2)
		int11 = integrate.simps(y11, self.nwl1, even='avg')
		int12 = integrate.simps(y12, self.nwl1, even='avg')
		int21 = integrate.simps(y21, self.nwl2, even='avg')
		int22 = integrate.simps(y22, self.nwl2, even='avg')
		self.cent1 = int11 / int12
		self.cent2 = int21 / int22
		self.becog = (self.cent1 - self.cent2) / (9.34*10**(-13) * ((self.cent1+self.cent2)/2.)**2 * glande)
		print("Line from %.4f to %.4f" %(self.wl[0], self.wl[-1]))
		print("Center of gravity: centers on %.4f           and %.4f          , Be = %.0f G" %(self.cent1, self.cent2, self.becog))
		# gaussian fit. Estimate initial parameters first
		par01 = max(self.cont1); par02 = max(self.cont2)
		par11 = max(self.r1); par12 = max(self.r2)
		par21 = self.cent1; par22 = self.cent2
		par31 = 0.4 * (self.nwl1[-1] - self.nwl1[0]); par32 = par31
		p1 = array([par01, 0., par11, par21, par31])
		p2 = array([par02, 0., par12, par22, par32])
		opt1, pcov1, infodict1, errmsg1, success1 = leastsq(errfunc, p1, args=(self.wl, self.r1), full_output=True)
		opt2, pcov2, infodict2, errmsg2, success2 = leastsq(errfunc, p2, args=(self.wl, self.r2), full_output=True)
		if pcov1 is not None and pcov2 is not None:
			s_sq1 = (errfunc(opt1, self.wl, self.r1)**2).sum()/(len(self.r1)-len(p1))
			pcov1 = pcov1 * s_sq1
			s_sq2 = (errfunc(opt2, self.wl, self.r2)**2).sum()/(len(self.r2)-len(p2))
			pcov2 = pcov2 * s_sq2
		else:
			pcov1 = np.inf
			pcov2 = np.inf
		errors1 = []; errors2 = []
		for i in range(len(opt1)):
			errors1.append(absolute(pcov1[i][i])**0.5)
			errors2.append(absolute(pcov2[i][i])**0.5)
		self.func1 = gauss(opt1, self.wl)
		self.func2 = gauss(opt2, self.wl)
		self.gcent1 = opt1[3]; self.gcent2 = opt2[3]
		fwhm1 = opt1[4]; fwhm2 = opt2[4]
		self.begauss = (self.gcent1 - self.gcent2) / (9.34*10**(-13) * ((self.gcent1+self.gcent2)/2.)**2 * glande)
		if errors1[3] <= 0.05 and errors2[3] <= 0.05:
			print("Gauss fit:         centers on %.4f (±%.4f) and %.4f (±%.4f), Be = %.0f G" %(self.gcent1, errors1[3], self.gcent2, errors2[3], self.begauss))
		else:
			print("Gauss fit:         centers on %.4f (±%.4f) and %.4f (±%.4f), Be = %.0f G (BIG UNCERTAINTY!)" %(self.gcent1, errors1[3], self.gcent2, errors2[3], self.begauss))
		self.res = vstack((self.cent1,self.cent2, int(self.becog), self.gcent1, self.gcent2, int(self.begauss), fwhm1, fwhm2, glande))



class Graphics(object):
	def __init__(self, wl, r1, r2):	# Initialize graphics using Matplotlib
		self.wli = wl
		self.ri1 = r1
		self.ri2 = r2
		self.fig1 = plt.figure(figsize=(15,5))
		self.ax1 = self.fig1.add_subplot(1,1,1)
		plt.subplots_adjust(bottom=0.35)
		self.ax1.set_xlabel('Wavelength, Angstroms')
		self.ax1.set_ylabel('Residual intensity')
		# Widgets
		# Button "Select line"
		axis_line = plt.axes([0.15, 0.025, 0.1, 0.04])
		self.button_line = Button(axis_line, 'Select line', color='grey', hovercolor='0.975')
		self.button_line.on_clicked(self.lineselect)
		# Button "Measure line"
		axis_measure = plt.axes([0.26, 0.025, 0.1, 0.04])
		self.button_measure = Button(axis_measure, 'Measure line', color='grey', hovercolor='0.975')
		self.button_measure.on_clicked(self.measure_line)
		# Button "Write result"
		axis_writeout = plt.axes([0.37, 0.025, 0.1, 0.04])
		self.button_writeout = Button(axis_writeout, 'Write results', color='green', hovercolor='0.975')
		self.button_writeout.on_clicked(self.write_line)
		# Button "Dump line"
		axis_dumpline = plt.axes([0.48, 0.025, 0.1, 0.04])
		self.button_dumpline = Button(axis_dumpline, 'Dump line', color='grey', hovercolor='0.975')
		self.button_dumpline.on_clicked(self.dump_line)
		# Button "Reset plot"
		axis_reset = plt.axes([0.63, 0.025, 0.1, 0.04])
		self.button_reset = Button(axis_reset, 'Reset plot', color='orange', hovercolor='0.975')
		self.button_reset.on_clicked(self.reset)
		# Button "Exit"
		axis_exit = plt.axes([0.74, 0.025, 0.1, 0.04])
		self.button_exit = Button(axis_exit, 'Exit app.', color='red', hovercolor='0.975')
		self.button_exit.on_clicked(exit)
		# Slider "line width"
		axis_width = plt.axes([0.15, 0.1, 0.65, 0.03])
		self.slider_width = Slider(axis_width, 'Selection Width', 0.5, 12.0, valinit=1.4)
		self.slider_width.on_changed(self.change_range)
		# Slider "line center"
		axis_shift = plt.axes([0.15, 0.20, 0.65, 0.03])
		self.slider_shift = Slider(axis_shift, 'Selection shift', -3., 3., valinit=0.)
		self.slider_shift.on_changed(self.change_range)
		# draw initial plot
		self.ax1.plot(self.wli, self.ri1, 'r-', lw=0.7)
		self.ax1.plot(self.wli, self.ri2, 'b-', lw=0.7)
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=0.5)
		plt.show()

	def lineselect(self, event):
		print("Mark center of measured line")
		self.cid1 = self.fig1.canvas.mpl_connect('button_press_event', self.selected)

	def selected(self, event):
		if (event.button == 1):
			yc = self.ax1.get_ylim()
			self.lc = event.xdata
			self.fig1.canvas.mpl_disconnect(self.cid1)
			width = self.slider_width.val
			shift = self.slider_shift.val
			center = (self.lc + shift)
			self.band, = self.ax1.bar(center, width=width, height=max(self.ri1), color='blue', alpha=0.3, align='center')
			self.ax1.set_ylim(yc[0], yc[1])
			plt.draw()
			self.line = ZLine()
			self.line.change_range(center-width/2., width)

	def change_range(self,event):
		width = self.slider_width.val
		shift = self.slider_shift.val
		center = self.lc + self.slider_shift.val
		self.band.set_width(width)
		self.band.set_x(center-width/2.)
		plt.draw()
		self.line.change_range(center-width/2., width)

	def measure_line(self, event):
		self.line.measure_line()
		self.ax1.plot(self.line.wl, self.line.func1, 'g-')
		self.ax1.plot(self.line.wl, self.line.func2, 'k-')
		mingfit = min(min(self.line.func1), min(self.line.func2))
		self.ax1.plot([self.line.gcent1, self.line.gcent1], [mingfit-0.03, mingfit-0.01], 'g-', lw=0.7)
		self.ax1.plot([self.line.gcent2, self.line.gcent2], [mingfit-0.03, mingfit-0.01], 'k-', lw=0.7)

		# self.ax1.plot(self.line.nwl1, self.line.cont1, 'r-') # draw semi-continuum
		# self.ax1.plot(self.line.nwl2, self.line.cont2, 'b-') # the same
		plt.draw()
		self.slider_shift.reset()

	def reset(self, event):
		self.ax1.clear()
		self.ax1.plot(self.wli, self.ri1, 'r-', lw=0.7)
		self.ax1.plot(self.wli, self.ri2, 'b-', lw=0.7)
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		self.slider_width.reset()
		self.slider_shift.reset()

	def write_line(self, event):
		global fh
		if hasattr(self.line, 'res'):
			savetxt(fh, self.line.res.transpose(), fmt='%10.4f')
			print("...saved")

	def dump_line(self, event):
		# Make text dump of lines
		outname = str(int(self.line.cent1))
		outarr1 = self.line.wl; outarr2 = self.line.wl
		savetxt(outname+'_1.line', vstack((outarr1, self.line.r1)).transpose(), fmt='%10.4f', delimiter='\t')
		savetxt(outname+'_2.line', vstack((outarr2, self.line.r2)).transpose(), fmt='%10.4f', delimiter='\t')



# Main block
if __name__ == "__main__":
	global fh
	if argv[1].find('_1.f') != -1:
		fh = open(argv[1].replace('_1.fits', '.res'), 'a')
	elif argv[1].find('_2.f') != -1:
		fh = open(argv[1].replace('_2.fits', '.res'), 'a')
	curtime = (datetime.now()).isoformat()
	fh.write('# ---- '+curtime+' ---- \n')
	fh.write('# λ1_cog    λ2_cog   Be_cog   λ1_gauss    λ2_gauss   Be_gauss  FWHM1   FWHM2   g_lande\n')
	spec = ZSpec()
	cnv = Graphics(spec.wli, spec.ri1, spec.ri2)
	close(fh)
	exit(0)
