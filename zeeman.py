#!/usr/local/bin/python3
#
# Programme aimed on measurements of zeeman shift in stellar polarized spectra
# by means of the center of gravity method and gaussian fitting of line's profile
# Author: Eugene Semenko. Last modification: 02 Feb 2021

from sys import exit, argv
import numpy as np
from astropy.io import fits
from scipy import interpolate, integrate
from scipy.optimize import curve_fit, leastsq
from scipy.stats import shapiro, norm
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from matplotlib.widgets import Button, Slider, Cursor

fontsize = 10
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times Palatino, New Century Schoolbook, Bookman,  Computer Modern Roman'

glande = 1.23
c = 299792.458 # speed of light in km/s
const1 = 9.34 * 10**(-13)
const2 = -4.67 * 10**(-13)

# Output data
Be_gs = np.array([])
Be_cog = np.array([])
fld_factor = np.array([])
v_stockes = np.array([])

# General functions
def interpol_spec(w, r, wl, d):
	tck = interpolate.splrep(w, r, s=0)
	ri = interpolate.splev(wl, tck, der=d)
	return ri

def contin_def(w, r1):
	meanx = np.mean(w)
	leftarr = r1[np.where((w >= w[0]) & (w <= meanx))]
	y1 = max(leftarr)
	x1 = w[np.where(r1 == y1)]
	rightarr = r1[np.where((w >= meanx) & (w <= w[-1]))]
	y2 = max(rightarr)
	x2 = w[np.where(r1 == y2)]
	xnew = w[np.where((w > x1) & (w < x2))]
	ynew = r1[np.where((w > x1) & (w < x2))]
	k = (y2 - y1) / (x2 - x1)
	b = y1 - k * x1
#	print("(%.4f, %.2f) - (%.4f, %.2f) k = %.3f, b = %.3f" %(x1, y1, x2, y2, k, b))
	return k * xnew + b, xnew, ynew

gauss = lambda p, x: p[0] + p[1] * x - p[2] * np.exp(-(4.*np.log2(2) * (x - p[3])**2)/ p[4]**2)

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
		try:
			hdu1 = fits.open(f1name)
			hdu2 = fits.open(f2name)
		except Exception as e:
			print(f"Error: {e}")
		hdr1 = hdu1[0].header; hdr2 = hdu2[0].header
		npix1 = hdr1['NAXIS1']; npix2 = hdr2['NAXIS1']
		crval1 = hdr1['CRVAL1']; crval2 = hdr2['CRVAL1']
		crpix1 = hdr1['CRPIX1']; crpix2 = hdr2['CRPIX1']
		cdelt1 = hdr1['CDELT1']; cdelt2 = hdr2['CDELT1']
		wl1 = crval1 + (np.arange(npix1) - crpix1) * cdelt1
		wl2 = crval2 + (np.arange(npix2) - crpix2) * cdelt2
		if hdr1['NAXIS'] == 3:
			r1 = hdu1[0].data[0, 0, :]
		elif hdr1['NAXIS'] == 1:
			r1 = hdu1[0].data
		else:
			r1 = hdu1[0].data[0]
		if hdr2['NAXIS'] == 3:
			r2 = hdu2[0].data[0, 0, :]
		elif hdr2['NAXIS'] == 1:
			r2 = hdu2[0].data
		else:
			r2 = hdu2[0].data[0]
		hdu1.close()
		hdu2.close()
		# Below are the class attributes
		self.wli = np.arange(max(wl1[0],wl2[0]), min(wl1[-1],wl2[-1]), cdelt1)
		self.ri1 = interpol_spec(wl1, r1, self.wli, 0)
		self.ri2 = interpol_spec(wl2, r2, self.wli, 0)
		self.istockes = (self.ri1 + self.ri2) / 2.0
		self.vstockes = (self.ri1 - self.ri2) / (self.ri1 + self.ri2)

	def analyse(self):
		global Be_gs, Be_cog, fld_factor, v_stockes
		mpl.rcParams['text.usetex'] = True
		# Classic method and its variations
		mean_be_cog = np.mean(Be_cog)
		sigma_be_cog = np.std(Be_cog, ddof=1)
		mean_be_gauss = np.mean(Be_gs)
		sigma_be_gauss = np.std(Be_gs, ddof=1)
		nl_cog = len(Be_cog)
		nl_gauss = len(Be_gs)
		nlines = np.max((nl_cog, nl_gauss))
		cog_d, cog_pval = shapiro(Be_cog)
		gauss_d, gauss_pval = shapiro(Be_gs)
		minbe = np.min((np.min(Be_cog), np.min(Be_gs)))
		maxbe = np.max((np.max(Be_cog), np.max(Be_gs)))
		with open('analysis_full.out', 'w') as fp:
			fp.write("===========================\n")
			fp.write("Classic positional methods:\n")
			fp.write("---------------------------------------------------------\n")
			fp.write(f"Be(COG)   = {mean_be_cog:.0f} ± {sigma_be_cog/np.sqrt(nl_cog):.0f} G (± {sigma_be_cog:.0f} G), {nl_cog} lines\n")
			fp.write(f"Be(Gauss)  = {mean_be_gauss:.0f} ± {sigma_be_gauss/np.sqrt(nl_gauss):.0f} G (± {sigma_be_gauss:.0f} G), {nl_gauss} lines\n")
			fp.write("---------------------------------------------------------\n")
			fp.write("Shapiro-Wilk normality test:\n")
			fp.write(f"COG: D = {cog_d:.4f}, p-value = {cog_pval:.4f}\n")
			fp.write(f"Gaussian fit: D = {gauss_d:.4f}, p-value = {gauss_pval:.4f}\n")
			fp.write("============================\n")
			fp.close()
		print("===========================")
		print("Classic positional methods:")
		print("---------------------------------------------------------")
		print(f"Be(COG)   = {mean_be_cog:.0f} ± {sigma_be_cog/np.sqrt(nl_cog):.0f} G (± {sigma_be_cog:.0f} G), {nl_cog} lines")
		print(f"Be(Gauss)  = {mean_be_gauss:.0f} ± {sigma_be_gauss/np.sqrt(nl_gauss):.0f} G (± {sigma_be_gauss:.0f} G), {nl_gauss} lines")
		print("---------------------------------------------------------")
		print("Shapiro-Wilk normality test:")
		print(f"COG: D = {cog_d:.4f}, p-value = {cog_pval:.4f}")
		print(f"Gaussian fit: D = {gauss_d:.4f}, p-value = {gauss_pval:.4f}")
		print("============================")
		# Regression
		N = len(fld_factor)
		det = N * np.sum(fld_factor**2) - (np.sum(fld_factor))**2
		a = ((np.sum(fld_factor**2) * np.sum(v_stockes)) - (np.sum(fld_factor) * np.sum(fld_factor*v_stockes))) / det
		b = ((N*np.sum(fld_factor*v_stockes)) - (np.sum(fld_factor)*np.sum(v_stockes))) / det
		sigma_y = np.sqrt(np.sum((v_stockes - a - b*fld_factor)**2) / (N - 2))
		sigma_a = np.sqrt(sigma_y**2 * np.sum(fld_factor**2) / det)
		sigma_b = np.sqrt(N * sigma_y**2 / det)
		chisq = np.sum(((v_stockes - a - b*fld_factor) / sigma_y)**2)
		print("======================")
		print("Regressional analysis:")
		print(f"<Bz> = {b:.0f} ± {sigma_b:.0f} G, χ^2 = {chisq:.2f} ({np.random.chisquare(N-2, 1)[0]:.2f})")
		print("======================")
		# Visualization. Classic
		if nlines >= 30:
			nbins = 10
		else:
			nbins = int(np.log2(nlines))+1
		fig_c = plt.figure(figsize=(8, 3), tight_layout=True)
		fig_c.subplots_adjust(top=0.96,bottom=0.15,right=0.98,left=0.08, hspace=0.4,   wspace=0.2)
		ax_c = fig_c.add_subplot(1,2,1)
		nn, bins, _ = ax_c.hist(Be_cog, nbins, color='b', density=True, histtype='bar', align='mid', alpha=0.7)
		pdf1 = norm.pdf(bins, mean_be_cog, sigma_be_cog)
		ax_c.plot(bins, pdf1, 'k--', linewidth=2)
		ax_c.set_xlabel(r"$B_\mathrm{e}$(COG), G",  fontsize=fontsize)
		ax_c.set_ylabel(r"Frequency", fontsize=fontsize)
		ax1_c = fig_c.add_subplot(1,2,2)
		nn, bins, _ = ax1_c.hist(Be_gs, nbins, color='r', density=True, histtype='bar', align='mid', alpha=0.7)
		pdf1 = norm.pdf(bins, mean_be_gauss, sigma_be_cog)
		ax1_c.plot(bins, pdf1, 'k--', linewidth=2)
		ax1_c.set_xlabel(r"$B_\mathrm{e}$(gaussian), G",  fontsize=fontsize)
		ax1_c.set_ylabel(r"Frequency", fontsize=11)
		# Visualization.Regression
		fig_r = plt.figure(figsize=(15,6))
		plt.subplots_adjust(hspace=0.45)
		ax1_r = fig_r.add_subplot(2,1,1)
		ax2_r = fig_r.add_subplot(2,1,2)
		ax1_r.set_title(rf"$<B_z> =\, {b:.0f}\,\pm\,{sigma_b:.0f}$ G, $\chi^2 =\, {chisq:.1f}$")
		ax1_r.set_xlabel(r'$-4.67\cdot10^{-13} g_{eff} \lambda^2 \frac{1}{I} \frac{\mathrm{d}I}{\mathrm{d}\lambda}$')
		ax1_r.set_ylabel(r'$V/I$')
		ax1_r.plot(fld_factor, v_stockes, 'bs')
		ax1_r.errorbar(fld_factor, v_stockes, yerr=sigma_y, ecolor='g', elinewidth=2, fmt="none", markeredgewidth=0)
		ax1_r.plot(fld_factor, b * fld_factor + a, 'r')
		ax2_r.set_xlabel(r"$\lambda,\, \mathrm{\AA}$")
		ax2_r.set_ylabel(r"$I,\, V/I\cdot5+1.05$")
		ax2_r.plot(self.wli, self.istockes, 'r-')
		ax2_r.plot(self.wli, self.vstockes*5 + 1.05, 'g')
		# draw histograms
		# formatter = FormatStrFormatter('%1.0e')
		mu_vsub = np.mean(v_stockes)
		sigma_vsub = np.std(v_stockes, ddof=1)
		mu_mfield = np.mean(fld_factor)
		sigma_mfield = np.std(fld_factor, ddof=1)
		fig2_r = plt.figure(figsize=(12,4))
		ax21_r = fig2_r.add_subplot(2,2,1)
		ax22_r = fig2_r.add_subplot(2,2,2)
		plt.subplots_adjust(wspace=0.45,bottom=-0.8)
		if len(v_stockes) <= 30:
			nsub = np.log2(len(v_stockes))+1
		else:
			nsub = 15
		ax21_r.set_xlabel(r'$V/I$')
		nn, bins, _ = ax21_r.hist(v_stockes, bins=nsub, density=True, histtype='bar', align='mid', color='g', alpha=0.7)
		pdf1 = norm.pdf(bins, mu_vsub, sigma_vsub)
		ax21_r.plot(bins, pdf1, 'r--', linewidth=2)
		ax22_r.set_xlabel(r'$-4.67\cdot10^{-13} g_{eff} \lambda^2 \frac{1}{I} \frac{\mathrm{d}I}{\mathrm{d}\lambda}$')
		# ax22_r.xaxis.set_major_formatter(formatter)
		nn, bins, _ = ax22_r.hist(fld_factor, bins=nsub, density=True, histtype='bar', align='mid', color='g', alpha=0.7)
		pdf2 = norm.pdf(bins, mu_mfield, sigma_mfield)
		ax22_r.plot(bins, pdf2, 'r--', linewidth=2)
		plt.show()


class ZLine(ZSpec):
	def __init__(self):
		self.wl = None
		self.r1 = None
		self.r2 = None
		self.vss = None
		self.iss = None

	def close(self):
		pass

	def change_range(self, wl0, width):
		self.wl = ZSpec().wli[np.where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		self.r1 = ZSpec().ri1[np.where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		self.r2 = ZSpec().ri2[np.where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		self.vss = (self.r1 - self.r2) / (self.r1 + self.r2)
		self.iss = (self.r1 + self.r2) / 2.

	def measure_line(self):
		global Be_gs, Be_cog, fld_factor, v_stockes
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
		self.becog = (self.cent1 - self.cent2) / (const1 * ((self.cent1+self.cent2)/2.)**2 * glande)
		print("Line from %.4f to %.4f" %(self.wl[0], self.wl[-1]))
		print("Center of gravity: centers on %.4f           and %.4f          , Be = %.0f G" %(self.cent1, self.cent2, self.becog))
		# Gaussian fit. Estimate initial parameters first
		par01 = max(self.cont1); par02 = max(self.cont2)
		par11 = max(self.r1); par12 = max(self.r2)
		par21 = self.cent1; par22 = self.cent2
		par31 = 0.4 * (self.nwl1[-1] - self.nwl1[0]); par32 = par31
		p1 = np.array([par01, 0., par11, par21, par31])
		p2 = np.array([par02, 0., par12, par22, par32])
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
			errors1.append(np.absolute(pcov1[i][i])**0.5)
			errors2.append(np.absolute(pcov2[i][i])**0.5)
		self.func1 = gauss(opt1, self.wl)
		self.func2 = gauss(opt2, self.wl)
		self.gcent1 = opt1[3]; self.gcent2 = opt2[3]
		fwhm1 = opt1[4]; fwhm2 = opt2[4]
		self.begauss = (self.gcent1 - self.gcent2) / (const1 * ((self.gcent1+self.gcent2)/2.)**2 * glande)
		if errors1[3] <= 0.05 and errors2[3] <= 0.05:
			print("Gauss fit:         centers on %.4f (±%.4f) and %.4f (±%.4f), Be = %.0f G" %(self.gcent1, errors1[3], self.gcent2, errors2[3], self.begauss))
		else:
			print("Gauss fit:         centers on %.4f (±%.4f) and %.4f (±%.4f), Be = %.0f G (BIG UNCERTAINTY!)" %(self.gcent1, errors1[3], self.gcent2, errors2[3], self.begauss))
		self.res = np.vstack((self.cent1,self.cent2, int(self.becog), self.gcent1, self.gcent2, int(self.begauss), fwhm1, fwhm2, glande))
		# Regressional analysis
		tck = interpolate.splrep(self.wl, self.iss, s=0)
		iss_der = interpolate.splev(self.wl, tck, der=1)
		fld_factor_loc = const2 * glande * self.wl**2 * (1./self.iss) * iss_der
		# Array of the results
		Be_gs = np.append(Be_gs, self.begauss)
		Be_cog = np.append(Be_cog, self.becog)
		fld_factor = np.append(fld_factor, fld_factor_loc)
		v_stockes = np.append(v_stockes, self.vss)


class Graphics(object):
	def __init__(self, wl, r1, r2):	# Initialize graphics using Matplotlib
		global mask
		self.wli = wl
		self.ri1 = r1
		self.ri2 = r2
		self.wl_msk = np.array([])
		self.dwl_msk = np.array([])
		self.fig1 = plt.figure(figsize=(15,6))
		self.ax1 = self.fig1.add_subplot(1,1,1)
		plt.subplots_adjust(bottom=0.35)
		self.ax1.set_xlabel('Wavelength, Angstroms')
		self.ax1.set_ylabel('Residual intensity')
		if mask == "":
			# Widgets
			# Button "Select line"
			axis_line = plt.axes([0.04, 0.025, 0.1, 0.04])
			self.button_line = Button(axis_line, 'Select line', color='grey', hovercolor='0.975')
			self.button_line.on_clicked(self.lineselect)
			# Button "Measure line"
			axis_measure = plt.axes([0.15, 0.025, 0.1, 0.04])
			self.button_measure = Button(axis_measure, 'Measure line', color='grey', hovercolor='0.975')
			self.button_measure.on_clicked(self.measure_line)
			# Button "Write result"
			axis_writeout = plt.axes([0.26, 0.025, 0.1, 0.04])
			self.button_writeout = Button(axis_writeout, 'Write results', color='green', hovercolor='0.975')
			self.button_writeout.on_clicked(self.write_line)
			# Button "Dump line"
			axis_dumpline = plt.axes([0.37, 0.025, 0.1, 0.04])
			self.button_dumpline = Button(axis_dumpline, 'Dump line', color='grey', hovercolor='0.975')
			self.button_dumpline.on_clicked(self.dump_line)
		# Button "Save mask"
		axis_savemask = plt.axes([0.48, 0.025, 0.1, 0.04])
		if mask == "":
			name = 'Save mask'
		else:
			name = 'Measure by mask'
		self.button_savemask = Button(axis_savemask, name, color='grey', hovercolor='0.975')
		if args.usemask == "":
			self.button_savemask.on_clicked(self.save_mask)
		else:
			self.button_savemask.on_clicked(self.measure_mask)
		# Button "Analyse"
		axis_analyse = plt.axes([0.59, 0.025, 0.1, 0.04])
		self.button_analyse = Button(axis_analyse, 'Analyse results', color='blue', hovercolor='0.975')
		self.button_analyse.on_clicked(self.analyse)
		# Button "Reset plot"
		axis_reset = plt.axes([0.74, 0.025, 0.1, 0.04])
		self.button_reset = Button(axis_reset, 'Reset plot', color='orange', hovercolor='0.975')
		self.button_reset.on_clicked(self.reset)
		# Button "Exit"
		axis_exit = plt.axes([0.85, 0.025, 0.1, 0.04])
		self.button_exit = Button(axis_exit, 'Exit app.', color='red', hovercolor='0.975')
		self.button_exit.on_clicked(exit)
		# draw initial plot
		self.ax1.plot(self.wli, self.ri1, 'r-', lw=0.7)
		self.ax1.plot(self.wli, self.ri2, 'b-', lw=0.7)
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		if mask != "":
			self.readin_mask(mask)
		cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=0.5)
		# Controls
		if mask != "":
			# Slider "Shift mask"
			axis_shiftmask = plt.axes([0.15, 0.10, 0.65, 0.03])
			self.slider_shiftmask = Slider(axis_shiftmask, 'Shift mask [km/s]', -80, 80, valinit=0)
			self.slider_shiftmask.on_changed(self.shiftmask)
		else:
			# Slider "line width"
			axis_width = plt.axes([0.15, 0.1, 0.65, 0.03])
			self.slider_width = Slider(axis_width, 'Selection Width', 0.5, 12.0, valinit=1.4)
			self.slider_width.on_changed(self.change_range)
			# Slider "line center"
			axis_shift = plt.axes([0.15, 0.20, 0.65, 0.03])
			self.slider_shift = Slider(axis_shift, 'Selection shift', -3., 3., valinit=0.)
			self.slider_shift.on_changed(self.change_range)
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
		global mask
		self.line.measure_line()
		self.ax1.plot(self.line.wl, self.line.func1, 'g-')
		self.ax1.plot(self.line.wl, self.line.func2, 'k-')
		mingfit = min(min(self.line.func1), min(self.line.func2))
		self.ax1.plot([self.line.gcent1, self.line.gcent1], [mingfit-0.03, mingfit-0.01], 'g-', lw=0.7)
		self.ax1.plot([self.line.gcent2, self.line.gcent2], [mingfit-0.03, mingfit-0.01], 'k-', lw=0.7)
		# self.ax1.plot(self.line.nwl1, self.line.cont1, 'r-') # draw semi-continuum
		# self.ax1.plot(self.line.nwl2, self.line.cont2, 'b-') # the same
		plt.draw()
		if mask == "":
			self.slider_shift.reset()

	def analyse(self, event):
		ZSpec().analyse()

	def measure_mask(self, event):
		for ln in range(len(self.wl_msk)):
			self.line = ZLine()
			self.line.change_range(self.wl_msk[ln]-self.dwl_msk[ln], 2 * self.dwl_msk[ln])
			self.measure_line(self)
			self.write_line(self)
		print("Measuring using the mask has completed.")

	def readin_mask(self, file_mask):
		self.wl_msk, self.dwl_msk = np.loadtxt(file_mask, unpack=True, usecols=(0,1), delimiter=';', comments='#')
		self.wl0_msk = self.wl_msk
		self.band_msk = self.ax1.bar(self.wl_msk, width=self.dwl_msk*2., height=max(self.ri1), color='orange', alpha=0.5, align='center')
		plt.draw()

	def shiftmask(self, event):
		yc = self.ax1.get_ylim()
		self.wl_msk = self.wl0_msk * np.sqrt((1. + self.slider_shiftmask.val/c)/(1. - self.slider_shiftmask.val/c))
		for i in range(len(self.wl_msk)):
			self.band_msk[i].set_x(self.wl_msk[i]-self.dwl_msk[i])
		plt.draw()
		self.ax1.set_ylim(yc[0], yc[1])

	def reset(self, event):
		global mask
		self.ax1.clear()
		self.ax1.plot(self.wli, self.ri1, 'r-', lw=0.7)
		self.ax1.plot(self.wli, self.ri2, 'b-', lw=0.7)
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		if mask == "":
			self.slider_width.reset()
			self.slider_shift.reset()
		else:
			self.slider_shiftmask.reset()

	def write_line(self, event):
		global fh
		if hasattr(self.line, 'res'):
			np.savetxt(fh, self.line.res.transpose(), fmt='%10.4f')
			self.line.close()
			print("...saved")
			self.ax1.text((self.line.gcent1 + self.line.gcent2)/2, min(min(self.line.func1), min(self.line.func2)) - 0.04, 'S')
			self.wl_msk = np.append(self.wl_msk, np.mean(self.line.wl))
			self.dwl_msk = np.append(self.dwl_msk, np.mean(self.line.wl)-self.line.wl[0])

	def save_mask(self, event):
		mask_name = "zeeman_gen.mask"
		output = np.zeros(self.wl_msk.size, dtype=[('wave', float), ('width', float), ('id', 'U32'), ('lande', float)])
		output['wave'] = self.wl_msk
		output['width'] = self.dwl_msk
		output['id'] = np.repeat('NoID', len(self.wl_msk))
		output['lande'] = 1.23 * np.ones(len(self.wl_msk))
		try:
			np.savetxt(mask_name, output, header='Wl0  ;  dWl  ;   ID   ;  g_lande', fmt="%.4f; %.4f; %s; %.2f")
		finally:
			print("Mask \'zeeman_gen.mask\' saved.")

	def dump_line(self, event):
		# Make text dump of lines
		outname = str(int(self.line.cent1))
		outarr1 = self.line.wl; outarr2 = self.line.wl
		np.savetxt(outname+'_1.line', np.vstack((outarr1, self.line.r1)).transpose(), fmt='%10.4f', delimiter='\t')
		np.savetxt(outname+'_2.line', np.vstack((outarr2, self.line.r2)).transpose(), fmt='%10.4f', delimiter='\t')


# Main block
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("spec", help="Input spectrum", type=str, default="")
	parser.add_argument("--usemask", help="Use mask of lines", type=str, default="")
	args = parser.parse_args()

	mask = args.usemask
	global fh

	if args.spec.find('_1.f') != -1:
		fh = open(args.spec.replace('_1.fits', '.res'), 'a')
	elif args.spec.find('_2.f') != -1:
		fh = open(args.spec.replace('_2.fits', '.res'), 'a')
	curtime = (datetime.now()).isoformat()
	fh.write('# ---- '+curtime+' ---- \n')
	fh.write('# λ1_cog    λ2_cog   Be_cog   λ1_gauss    λ2_gauss   Be_gauss  FWHM1   FWHM2   g_lande\n')
	spec = ZSpec()
	cnv = Graphics(spec.wli, spec.ri1, spec.ri2)
	fh.close()
	exit(0)
