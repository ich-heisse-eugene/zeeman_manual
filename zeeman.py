#!/usr/local/bin/python3.11
#
# Programme aimed on measurements of zeeman shift in stellar polarized spectra
# by means of the center of gravity method and gaussian fitting of line's profile
# Author: Eugene Semenko. Last modification: 04 Oct 2024

from sys import exit, argv
import os
import numpy as np
from astropy.io import fits
from scipy import interpolate, integrate
from scipy.optimize import curve_fit, leastsq
from scipy.stats import shapiro, norm
from scipy.special import erf
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from matplotlib.widgets import Button, Slider, Cursor

fontsize = 10
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.serif'] = ['Palatino, Latin Modern Roman']
mpl.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']

glande = 1.23
c = 299792.458 # speed of light in km/s
const1 = 9.34 * 10**(-13)
const2 = -4.67 * 10**(-13)

# Output data
Be_gs = np.array([])
Be_cog = np.array([])
fld_factor = np.array([])
v_stockes = np.array([])
bs_v = np.array([])
bs_fl = np.array([])
bs_fld = np.array([])
v_ref = np.array([])

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

gauss = lambda p, x: p[0] - p[1] * np.exp(-(4.*np.log2(2) * (x - p[2])**2)/ p[3]**2)

errfunc = lambda p, x, y: y - gauss(p, x)

# def gauss(p, x):
# 	slope = p[0] + p[1] * x
# 	norm = p[2] * np.exp(-(4.*np.log2(2) * (x - p[3])**2)/ p[4]**2)
# 	skew = 1 + erf(np.sqrt(8 * np.log2(2))*p[5] * (x - p[3])/(np.sqrt(2)*p[4]))
# 	return slope - norm*skew

def bisectors(wl, par, r1, r2, v0, v0r1, v0r2):
	left_wing_w = wl[wl <= v0]
	right_wing_w = wl[wl >= v0]
	left_wing_i = par[wl <= v0]
	right_wing_i = par[wl >= v0]
	left_wing_r1 = r1[wl <= v0r1]
	right_wing_r1 = r1[wl >= v0r1]
	left_wing_r2 = r2[wl <= v0r2]
	right_wing_r2 = r2[wl >= v0r2]
	flux_bs = np.linspace(np.max([np.min(left_wing_i), np.min(right_wing_i)]), np.min([np.max(left_wing_i), np.max(right_wing_i)]), len(left_wing_i) + len(right_wing_i))
	if len(left_wing_i) > 2 or len(right_wing_i) > 2:
		try:
			left_wing_wi = interpol_spec(np.flip(left_wing_i), np.flip(left_wing_w), flux_bs, 0)
			right_wing_wi = interpol_spec(right_wing_i, right_wing_w, flux_bs, 0)
			left_wing_wr1 = interpol_spec(np.flip(left_wing_r1), np.flip(left_wing_w), flux_bs, 0)
			right_wing_wr1 = interpol_spec(right_wing_r1, right_wing_w, flux_bs, 0)
			left_wing_wr2 = interpol_spec(np.flip(left_wing_r2), np.flip(left_wing_w), flux_bs, 0)
			right_wing_wr2 = interpol_spec(right_wing_r2, right_wing_w, flux_bs, 0)
			bs_i = (left_wing_wi + right_wing_wi) / 2.
			bs_r1 = (left_wing_wr1 + right_wing_wr1) / 2.
			bs_r2 = (left_wing_wr2 + right_wing_wr2) / 2.
			return (c * (bs_i - v0))/v0, flux_bs, bs_r1, bs_r2, bs_i
		except Exception as e:
			# print(f"Error while deriving bisectors: {e}")
			return None, None, None, None, None


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
		global Be_gs, Be_cog, fld_factor, v_stockes, bs_v, bs_fl, bs_fld, v_ref
		global report_fh
		global argsbs
		mpl.rcParams['text.usetex'] = True
		# Classic method and its variations
		for i in range(5):
			idx_gs = np.where((Be_gs >= np.mean(Be_gs) - 3.*np.std(Be_gs, ddof=1)) & (Be_gs <= np.mean(Be_gs) + 3.*np.std(Be_gs, ddof=1)))
			idx_cog = np.where((Be_cog >= np.mean(Be_cog) - 3.*np.std(Be_cog, ddof=1)) & (Be_cog <= np.mean(Be_cog) + 3.*np.std(Be_cog, ddof=1)))
			Be_cog = Be_cog[idx_cog]
			Be_gs = Be_gs[idx_gs]
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
		with open(report_fh, 'w') as fp:
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
		with open(report_fh, 'a') as fp:
			fp.write("======================\n")
			fp.write("Regressional analysis:\n")
			fp.write(f"<Bz> = {b:.0f} ± {sigma_b:.0f} G, χ^2 = {chisq:.2f} ({np.random.chisquare(N-2, 1)[0]:.2f})\n")
			fp.write("======================\n")
			fp.close()
		print("======================")
		print("Regressional analysis:")
		print(f"<Bz> = {b:.0f} ± {sigma_b:.0f} G, χ^2 = {chisq:.2f} ({np.random.chisquare(N-2, 1)[0]:.2f})")
		print("======================")
		fp.close()
		# Visualization. Classic
		if nlines >= 30:
			nbins = 10
		else:
			nbins = int(np.log2(nlines))+1
		fig_c = plt.figure(figsize=(8, 3), tight_layout=True)
		fig_c.subplots_adjust(top=0.96,bottom=0.15,right=0.98,left=0.08, hspace=0.4,   wspace=0.2)
		ax_c = fig_c.add_subplot(1,2,1)
		nn, bins, _ = ax_c.hist(Be_cog, nbins, color='#75bbfd', density=True, histtype='bar', align='mid', alpha=0.7)
		pdf1 = norm.pdf(bins, mean_be_cog, sigma_be_cog)
		ax_c.plot(bins, pdf1, 'k--', linewidth=2)
		ax_c.set_xlabel(r"$B_\mathrm{e}$(COG), G",  fontsize=fontsize)
		ax_c.set_ylabel(r"Frequency", fontsize=fontsize)
		ax1_c = fig_c.add_subplot(1,2,2)
		nn, bins, _ = ax1_c.hist(Be_gs, nbins, color='#ff796c', density=True, histtype='bar', align='mid', alpha=0.7)
		pdf1 = norm.pdf(bins, mean_be_gauss, sigma_be_cog)
		ax1_c.plot(bins, pdf1, 'k--', linewidth=2)
		ax1_c.set_xlabel(r"$B_\mathrm{e}$(gaussian), G",  fontsize=fontsize)
		ax1_c.set_ylabel(r"Frequency", fontsize=11)
		fig_c.savefig(report_fh+".positional.distrib.pdf", dpi=250)
		# Visualization.Regression
		fig_r = plt.figure(figsize=(15,6))
		plt.subplots_adjust(hspace=0.45)
		ax1_r = fig_r.add_subplot(2,1,1)
		ax1_r.tick_params(axis='both', direction='in')
		ax1_r.xaxis.set_ticks_position('both')
		ax1_r.yaxis.set_ticks_position('both')
		ax2_r = fig_r.add_subplot(2,1,2)
		ax2_r.tick_params(axis='both', direction='in')
		ax2_r.xaxis.set_ticks_position('both')
		ax2_r.yaxis.set_ticks_position('both')
		ax1_r.set_title(rf"$<B_z> =\, {b:.0f}\,\pm\,{sigma_b:.0f}$ G, $\chi^2 =\, {chisq:.1f}$")
		ax1_r.set_xlabel(r'$-4.67\cdot10^{-13} g_{eff} \lambda^2 \frac{1}{I} \frac{\mathrm{d}I}{\mathrm{d}\lambda}$')
		ax1_r.set_ylabel(r'$V/I$')
		ax1_r.plot(fld_factor, v_stockes, marker='s', ms=0.8, color='#5cac2d', ls='')
		ax1_r.errorbar(fld_factor, v_stockes, yerr=sigma_y, ecolor='#96ae8d', elinewidth=0.8, fmt="none", markeredgewidth=0)
		ax1_r.plot(fld_factor, b * fld_factor + a, color='#0504aa', lw=1, ls='-')
		ax2_r.set_xlabel(r"$\lambda,\, \mathrm{\AA}$")
		ax2_r.set_ylabel(r"$I,\, V/I\cdot5+1.05$")
		ax2_r.plot(self.wli, self.istockes, ls='-', color='#be0119', lw=0.8)
		ax2_r.plot(self.wli, self.vstockes*5 + 1.05, ls='-', color='#5cac2d', lw=0.8)
		# draw histograms
		# formatter = FormatStrFormatter('%1.0e')
		mu_vsub = np.mean(v_stockes)
		sigma_vsub = np.std(v_stockes, ddof=1)
		mu_mfield = np.mean(fld_factor)
		sigma_mfield = np.std(fld_factor, ddof=1)
		fig_r.savefig(report_fh+".regression.pdf", dpi=350)
		fig2_r = plt.figure(figsize=(12,4))
		ax21_r = fig2_r.add_subplot(2,2,1)
		ax22_r = fig2_r.add_subplot(2,2,2)
		plt.subplots_adjust(wspace=0.45,bottom=-0.8)
		if len(v_stockes) <= 30:
			nsub = np.log2(len(v_stockes))+1
		else:
			nsub = 15
		ax21_r.set_xlabel(r'$V/I$')
		nn, bins, _ = ax21_r.hist(v_stockes, bins=nsub, density=True, histtype='bar', align='mid', color='#39ad48', alpha=0.7)
		pdf1 = norm.pdf(bins, mu_vsub, sigma_vsub)
		ax21_r.plot(bins, pdf1, 'r--', linewidth=2)
		ax22_r.set_xlabel(r'$-4.67\cdot10^{-13} g_{eff} \lambda^2 \frac{1}{I} \frac{\mathrm{d}I}{\mathrm{d}\lambda}$')
		# ax22_r.xaxis.set_major_formatter(formatter)
		nn, bins, _ = ax22_r.hist(fld_factor, bins=nsub, density=True, histtype='bar', align='mid', color='#39ad48', alpha=0.7)
		pdf2 = norm.pdf(bins, mu_mfield, sigma_mfield)
		ax22_r.plot(bins, pdf2, 'r--', linewidth=2)
		fig2_r.savefig(report_fh+".regression.distrib.pdf", dpi=250)
		# Visualization. Bisectors
		if argsbs:
			idx_filter = np.where((bs_fld >= (np.mean(bs_fld) - 5*sigma_be_gauss)) & (bs_fld <= (np.mean(bs_fld) + 5*sigma_be_gauss)))
			bs_v = bs_v[idx_filter]
			bs_fl = bs_fl[idx_filter]
			bs_fld = bs_fld[idx_filter]
			v_ref = v_ref[idx_filter]
			fig_b = plt.figure(figsize=(10,6))
			plt.subplots_adjust(hspace=0.45)
			ax1_b = fig_b.add_subplot(2,1,1)
			ax2_b = fig_b.add_subplot(2,1,2)
			ax1_b.set_title(rf"Longitudinal field derived from bisectors")
			ax1_b.set_xlabel(r'Velocity, km\,s$^{-1}$')
			ax1_b.set_ylabel(r'$\langle B_{l} \rangle$, G')
			ax2_b.set_xlabel(r'Relative flux')
			ax2_b.set_ylabel(r'$\langle B_{l} \rangle$, G')
			ax1_b.plot(bs_v, bs_fld, 'b.', ms=2.8)
			ax2_b.plot(bs_fl, bs_fld, 'r.', ms=2.8)
			with open(report_fh+".bs", 'w') as fp:
				fp.write("# ======================\n")
				fp.write("# Bisectors:\n")
				fp.write("# V(km/s)      Flux      Bz      lam0(A):\n")
				fp.write("# ======================\n")
				for i in range(len(bs_fld)):
					fp.write(f"{bs_v[i]:.2f}\t{bs_fl[i]:.2f}\t{bs_fld[i]:.0f}\t{v_ref[i]:.4f}\n")
				fp.close()
			fig_b.savefig(report_fh+".bissectors.pdf", dpi=350)
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
		wl_t = ZSpec().wli[np.where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		r1_t = ZSpec().ri1[np.where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		r2_t = ZSpec().ri2[np.where((ZSpec().wli >= wl0) & (ZSpec().wli <= wl0+width))]
		iss_t = (r1_t + r2_t) / 2.
		ileft = np.argmax(iss_t[0:np.argmin(iss_t)+1])
		iright = np.argmin(iss_t) + np.argmax(iss_t[np.argmin(iss_t):len(iss_t)+2])
		self.wl = wl_t[ileft: iright+1]
		self.r1 = r1_t[ileft: iright+1]
		self.r2 = r2_t[ileft: iright+1]
		self.iss = iss_t[ileft: iright+1]
		self.vss = (self.r1 - self.r2) / (self.r1 + self.r2)

	def measure_line(self):
		global Be_gs, Be_cog, fld_factor, v_stockes, bs_v, bs_fl, bs_fld, v_ref
		global argsbs
		# Measure shift by COG method
		# Redetermination of continuum level
		self.cont1, self.nwl1, nr1 = contin_def(self.wl, self.r1)
		self.cont2, self.nwl2, nr2 = contin_def(self.wl, self.r2)
		cont, wlc, nrc = contin_def(self.wl, self.iss)
		if len(self.cont1) != 0 and len(self.cont2) != 0:
			y11 = self.nwl1 * (self.cont1 - nr1)
			y12 = (self.cont1 - nr1)
			y21 = self.nwl2 * (self.cont2 - nr2)
			y22 = (self.cont2 - nr2)
			try:
				int11 = integrate.simps(y11, self.nwl1, even='avg')
				int12 = integrate.simps(y12, self.nwl1, even='avg')
				int21 = integrate.simps(y21, self.nwl2, even='avg')
				int22 = integrate.simps(y22, self.nwl2, even='avg')
				if int12 != 0 and int22 != 0 and ~np.isnan(int21) and ~np.isnan(int22):
					self.cent1 = int11 / int12
					self.cent2 = int21 / int22
					self.becog = (self.cent1 - self.cent2) / (const1 * ((self.cent1+self.cent2)/2.)**2 * glande)
					print(f"Line spans from {self.wl[0]:.4f} to {self.wl[-1]:.4f}")
					print(f"Center-of-gravity: centers at {self.cent1:9.4f} Å         and {self.cent2:9.4f} Å,         Be = {self.becog:.0f} G")
				else:
					self.becog = -99999999
					self.cent1 = -99999999; self.cent2 = -99999999;
			except:
				self.becog = -99999999
				self.cent1 = -99999999; self.cent2 = -99999999;
			# Gaussian fit. Estimate initial parameters first
			par01 = max(self.cont1); par02 = max(self.cont2)
			par11 = max(self.r1); par12 = max(self.r2)
			if self.cent1 == -99999999 and self.cent2 != -99999999:
				par21 = self.cent2
			elif self.cent1 != -99999999 and self.cent2 == -99999999:
				par22 = self.cent1
			else:
				par21 = self.cent1; par22 = self.cent2
			par31 = 0.4 * (self.nwl1[-1] - self.nwl1[0]); par32 = par31
			# I parameter
			par001 = max(cont)
			par002 = max(self.iss)
			par003 = (self.cent1 + self.cent2) / 2.
			par004 = 0.4 * (wlc[-1] - wlc[0])
			p00 = np.array([par001, par002, par003, par004])
			try:
				opt0, pcov0, infodict0, errmsg0, success0 = leastsq(errfunc, p00, args=(self.wl, self.iss), maxfev=10000, full_output=True)
				self.gcent0 = opt0[2]
				self.depth0 = opt0[1]
				print(f"Line is centered @ {self.gcent0:.4f} Å")
			except Exception as e:
				print(f"Cannot fit I parameter: {e}")
			#
			p1 = np.array([par01, par11, par21, par31])
			p2 = np.array([par02, par12, par22, par32])
			opt1, pcov1, infodict1, errmsg1, success1 = leastsq(errfunc, p1, args=(self.wl, self.r1), maxfev=10000, full_output=True)
			opt2, pcov2, infodict2, errmsg2, success2 = leastsq(errfunc, p2, args=(self.wl, self.r2), maxfev=10000, full_output=True)
			if pcov1 is not None and pcov2 is not None:
				s_sq1 = (errfunc(opt1, self.wl, self.r1)**2).sum()/(len(self.r1)-len(p1))
				pcov1 = pcov1 * s_sq1
				s_sq2 = (errfunc(opt2, self.wl, self.r2)**2).sum()/(len(self.r2)-len(p2))
				pcov2 = pcov2 * s_sq2
				errors1 = []; errors2 = []
				for i in range(len(opt1)):
					errors1.append(np.absolute(pcov1[i][i])**0.5)
					errors2.append(np.absolute(pcov2[i][i])**0.5)
				self.fit_wl = np.linspace(self.wl[0], self.wl[-1], len(self.wl)*10)
				self.func1 = gauss(opt1, self.fit_wl)
				self.func2 = gauss(opt2, self.fit_wl)
				self.gcent1 = opt1[2]; self.gcent2 = opt2[2]
				fwhm1 = opt1[3]; fwhm2 = opt2[3]
				self.begauss = (self.gcent1 - self.gcent2) / (const1 * ((self.gcent1+self.gcent2)/2.)**2 * glande)
				if errors1[2] <= 0.05 and errors2[2] <= 0.05:
					print("Gauss fit:         centers at %.4f (±%.4f) and %.4f (±%.4f), Be = %.0f G" %(self.gcent1, errors1[2], self.gcent2, errors2[2], self.begauss))
				else:
					print("Gauss fit:         centers at %.4f (±%.4f) and %.4f (±%.4f), Be = %.0f G (BIG UNCERTAINTY!)" %(self.gcent1, errors1[2], self.gcent2, errors2[2], self.begauss))
			else:
				self.cent1 = -99999999; self.cent2 = -99999999
				self.becog = -99999999; self.begauss = -99999999
				self.gcent1 = -99999999; self.gcent2 = -99999999
				fwhm1 = -99999999; fwhm2 = -99999999
			self.shift_cog = (self.cent1 - self.cent2)
			self.shift_g = self.gcent1 - self.gcent2
			self.res = np.vstack((self.cent1,self.cent2, self.shift_cog, int(self.becog), self.gcent1, self.gcent2, self.shift_g, int(self.begauss), fwhm1, fwhm2, self.gcent0, self.depth0, glande))
		else:
			self.res = -1
		# Analysis of bisectors
		if args.bs:
			if len(self.cont1) != 0 and len(self.cont2) != 0 and self.gcent1 != -99999999 and self.gcent2 != -99999999:
				if np.min([self.iss[0], self.iss[-1]])/np.max([self.iss[0], self.iss[-1]]) >= 0.7 and \
			   	   np.min(self.iss)/np.min([self.iss[0], self.iss[-1]]) <= 0.9:
				   bs, bs_flux, bs1_w, bs2_w, bs_w = bisectors(self.wl, self.iss, self. r1, self.r2, self.gcent0, self.gcent1, self.gcent2)
				   if (bs is not None):
					   be_bs = (bs1_w - bs2_w) / (const1 * self.gcent0**2 * glande)
					   idx_sort = np.argsort(bs)
					   bs_fld = np.append(bs_fld, be_bs[idx_sort])
					   bs_fl = np.append(bs_fl, bs_flux[idx_sort])
					   bs_v = np.append(bs_v, bs[idx_sort])
					   v_ref = np.append(v_ref, np.repeat(self.gcent0, len(bs)))
					   if np.isfinite(self.gcent0):
						   print(f"Bisectors: line {self.gcent0:.3f} has {len(bs)} points")
						   output_line = "line_"+str(f"{self.gcent0:.1f}.dat")
						   output_bs = "bs_"+str(f"{self.gcent0:.1f}.dat")
						   np.savetxt("bisectors" + os.path.sep + output_line, np.vstack((self.wl, self.iss, self.r1, self.r2)).transpose(), fmt="%.4f", header="wl  I  r1  r2")
						   np.savetxt("bisectors" + os.path.sep + output_bs, np.vstack((bs, bs_flux, bs1_w, bs2_w, bs_w)).transpose(), fmt="%.4f", header="V(km/s)  Flux   lbs_r1, lbs_r2, lbs_i")
		# Regressional analysis
		if len(self.wl) > 3:
			tck = interpolate.splrep(self.wl, self.iss, s=0)
			iss_der = interpolate.splev(self.wl, tck, der=1)
			fld_factor_loc = const2 * glande * self.wl**2 * (1./self.iss) * iss_der
			fld_factor = np.append(fld_factor, fld_factor_loc)
			v_stockes = np.append(v_stockes, self.vss)
		else:
			print("The line has wrong length. Change the width and repeat measurements")
		# Array of the results
		if hasattr(self, 'begauss') and hasattr(self, 'becog'):
			Be_gs = np.append(Be_gs, self.begauss)
			Be_cog = np.append(Be_cog, self.becog)


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
		self.ax1.tick_params(axis='both', direction='in')
		self.ax1.xaxis.set_ticks_position('both')
		self.ax1.yaxis.set_ticks_position('both')
		self.lc = 0
		plt.subplots_adjust(bottom=0.35)
		self.ax1.set_xlabel('Wavelength, Angstroms')
		self.ax1.set_ylabel('Residual intensity')
		if mask == "":
			# Widgets
			# Button "Select line"
			axis_line = plt.axes([0.04, 0.025, 0.1, 0.04])
			self.button_line = Button(axis_line, 'Select line', color='#d8dcd6', hovercolor='0.975')
			self.button_line.on_clicked(self.lineselect)
			# Button "Measure line"
			axis_measure = plt.axes([0.15, 0.025, 0.1, 0.04])
			self.button_measure = Button(axis_measure, 'Measure line', color='#d8dcd6', hovercolor='0.975')
			self.button_measure.on_clicked(self.measure_line)
			# Button "Write result"
			axis_writeout = plt.axes([0.26, 0.025, 0.1, 0.04])
			self.button_writeout = Button(axis_writeout, 'Write results', color='#a6c875', hovercolor='0.975')
			self.button_writeout.on_clicked(self.write_line)
			# Button "Dump line"
			axis_dumpline = plt.axes([0.37, 0.025, 0.1, 0.04])
			self.button_dumpline = Button(axis_dumpline, 'Dump line', color='#d8dcd6', hovercolor='0.975')
			self.button_dumpline.on_clicked(self.dump_line)
		# Button "Save mask"
		axis_savemask = plt.axes([0.48, 0.025, 0.1, 0.04])
		if mask == "":
			name = 'Save mask'
		else:
			name = 'Measure by mask'
		self.button_savemask = Button(axis_savemask, name, color='#d8dcd6', hovercolor='0.975')
		if args.usemask == "":
			self.button_savemask.on_clicked(self.save_mask)
		else:
			self.button_savemask.on_clicked(self.measure_mask)
		# Button "Analyse"
		axis_analyse = plt.axes([0.59, 0.025, 0.1, 0.04])
		self.button_analyse = Button(axis_analyse, 'Analyse results', color='#75bbfd', hovercolor='0.975')
		self.button_analyse.on_clicked(self.analyse)
		# Button "Reset plot"
		axis_reset = plt.axes([0.74, 0.025, 0.1, 0.04])
		self.button_reset = Button(axis_reset, 'Reset plot', color='#fdaa48', hovercolor='0.975')
		self.button_reset.on_clicked(self.reset)
		# Button "Exit"
		axis_exit = plt.axes([0.85, 0.025, 0.1, 0.04])
		self.button_exit = Button(axis_exit, 'Exit app.', color='#ff474c', hovercolor='0.975')
		self.button_exit.on_clicked(self.exit)
		# draw initial plot
		self.ax1.plot(self.wli, self.ri1, linestyle='-', lw=1.1, marker='.', ms=1.2, color='#fc5a50')
		self.ax1.plot(self.wli, self.ri2, linestyle='-', lw=1, marker='.', ms=1.2, color='#069af3')
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		if mask != "":
			self.readin_mask(mask)
		cursor = Cursor(self.ax1, useblit=True, color='#e50000', linewidth=0.5)
		# Controls
		if mask != "":
			# Slider "Shift mask"
			axis_shiftmask = plt.axes([0.15, 0.10, 0.65, 0.03])
			self.slider_shiftmask = Slider(axis_shiftmask, 'Shift mask [km/s]', -150, 150, valinit=0)
			self.slider_shiftmask.on_changed(self.shiftmask)
		else:
			# Slider "line width"
			axis_width = plt.axes([0.15, 0.1, 0.65, 0.03])
			self.slider_width = Slider(axis_width, 'Selection Width', 0.5, 22.0, valinit=1.4)
			self.slider_width.on_changed(self.change_range)
			# Slider "line center"
			axis_shift = plt.axes([0.15, 0.20, 0.65, 0.03])
			self.slider_shift = Slider(axis_shift, 'Selection shift', -6., 6., valinit=0.)
			self.slider_shift.on_changed(self.change_range)
		plt.show()


	def lineselect(self, event):
		self.slider_shift.reset()
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
			self.band, = self.ax1.bar(center, width=width, height=max(self.ri1), color='#929591', alpha=0.3, align='center')
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
		if hasattr(self.line, 'func1') and hasattr(self.line, 'func2'):
			self.ax1.plot(self.line.fit_wl, self.line.func1, ls='-', color='#8f1402', lw=1)
			self.ax1.plot(self.line.fit_wl, self.line.func2, ls='-', color='#010fcc', lw=1)
			mingfit = min(min(self.line.func1), min(self.line.func2))
			self.ax1.plot([self.line.gcent1, self.line.gcent1], [mingfit-0.03, mingfit-0.01], ls='-', color='#8f1402', lw=0.7)
			self.ax1.plot([self.line.gcent2, self.line.gcent2], [mingfit-0.03, mingfit-0.01], ls='-', color='#010fcc', lw=0.7)
			# self.ax1.plot(self.line.nwl1, self.line.cont1, 'r-') # draw semi-continuum
			# self.ax1.plot(self.line.nwl2, self.line.cont2, 'b-') # the same
			plt.draw()

	def analyse(self, event):
		global report_fh
		ZSpec().analyse()

	def measure_mask(self, event):
		for ln in range(len(self.wl_msk)):
			self.line = ZLine()
			self.line.change_range(self.wl_msk[ln]-self.dwl_msk[ln], 2 * self.dwl_msk[ln])
			self.measure_line(self)
			if hasattr(self.line, 'func1') and hasattr(self.line, 'func2'):
				self.write_line(self)
		print("Measuring using the mask has completed.")

	def readin_mask(self, file_mask):
		self.wl_msk, self.dwl_msk = np.loadtxt(file_mask, unpack=True, usecols=(0,1), delimiter=';', comments='#')
		self.wl0_msk = self.wl_msk
		self.band_msk = self.ax1.bar(self.wl_msk, width=self.dwl_msk*2., height=max(self.ri1), color='orange', alpha=0.3, align='center')
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
		self.ax1.plot(self.wli, self.ri1, lw=1.1, marker='.', ms=1.2, color='#fc5a50')
		self.ax1.plot(self.wli, self.ri2, lw=1.1, marker='.', ms=1.2, color='#069af3')
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		if mask == "":
			self.slider_width.reset()
			self.slider_shift.reset()
		else:
			self.slider_shiftmask.reset()

	def write_line(self, event):
		global fh
		if len(self.line.res) > 1:
			np.savetxt(fh, self.line.res.transpose(), fmt='%10.4f')
			self.line.close()
			print("...saved")
			self.ax1.text((self.line.gcent1 + self.line.gcent2)/2, min(min(self.line.func1), min(self.line.func2)) - 0.06, 'S', ha='center')
			self.wl_msk = np.append(self.wl_msk, np.mean(self.line.wl))
			self.dwl_msk = np.append(self.dwl_msk, np.mean(self.line.wl)-self.line.wl[0])
		else:
			print("...skipped")

	def save_mask(self, event):
		global mask_fh
		output = np.zeros(self.wl_msk.size, dtype=[('wave', float), ('width', float), ('id', 'U32'), ('lande', float)])
		output['wave'] = self.wl_msk
		output['width'] = self.dwl_msk
		output['id'] = np.repeat('NoID', len(self.wl_msk))
		output['lande'] = 1.23 * np.ones(len(self.wl_msk))
		try:
			np.savetxt(mask_fh, output, header='Wl0  ;  dWl  ;   ID   ;  g_lande', fmt="%.4f; %.4f; %s; %.2f")
		finally:
			print(f"Mask {mask_fh} saved.")

	def dump_line(self, event):
		# Make text dump of lines
		outname = str(int(self.line.cent1))
		outarr1 = self.line.wl; outarr2 = self.line.wl
		np.savetxt(outname+'_1.line', np.vstack((outarr1, self.line.r1)).transpose(), fmt='%10.4f', delimiter='\t')
		np.savetxt(outname+'_2.line', np.vstack((outarr2, self.line.r2)).transpose(), fmt='%10.4f', delimiter='\t')

	def exit(self, event):
		self.ax1.set_xlim([self.wli[0]-1., self.wli[-1]+1.])
		self.ax1.set_ylim([0, 1.1])
		self.fig1.savefig(report_fh+".visual.pdf", dpi=350)
		exit(0)


# Main block
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("spec", help="Input spectrum", type=str, default="")
	parser.add_argument("--usemask", help="Use mask of lines", type=str, default="")
	parser.add_argument("--bs", help="Measure bisectors of the line", action="store_true")
	args = parser.parse_args()

	mask = args.usemask
	argsbs = args.bs
	global fh
	global mask_fh
	global report_fh

	if args.spec.find('_1.f') != -1:
		fh = open(args.spec.replace('_1.fits', '.res'), 'a')
		mask_fh = args.spec.replace('_1.fits', '.mask')
		report_fh = args.spec.replace('_1.fits', '.report')
	elif args.spec.find('_2.f') != -1:
		fh = open(args.spec.replace('_2.fits', '.res'), 'a')
		mask_fh = args.spec.replace('_2.fits', '.mask')
		report_fh = args.spec.replace('_2.fits', '.report')
	curtime = (datetime.now()).isoformat()
	fh.write('# ---- '+curtime+' ---- \n')
	fh.write('# λ1_cog    λ2_cog Δλ_cog  Be_cog   λ1_gauss    λ2_gauss Δλ_gauss  Be_gauss  FWHM1   FWHM2  λ0  r0  g_lande\n')
	spec = ZSpec()
	if argsbs:
		try:
			os.mkdir("bisectors")
		except Exception:
			pass
	cnv = Graphics(spec.wli, spec.ri1, spec.ri2)
	fh.close()
	exit(0)
