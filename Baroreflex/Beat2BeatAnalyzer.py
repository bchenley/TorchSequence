import numpy as np
from sklearn.cluster import KMeans

import scipy as sc

import matplotlib.pyplot as plt

import importlib

from Beat2BeatAnalysis import butter, periodogram, moving_average, Interpolator

class Beat2BeatAnalyzer():
  def __init__(self, dt, ecg, abp):

    self.dt = dt
    self.ecg, self.abp = ecg, abp
    self.t = np.arange(len(ecg))*self.dt

  def fill(self, interp_kind = 'linear'):

    interpolator = Interpolator(kind = interp_kind)

    if np.any(np.isnan(self.ecg)):

      ecg_notnan = self.ecg[~np.isnan(self.ecg)]
      t_notnan = self.t[~np.isnan(self.ecg)]

      interpolator.fit(t_notnan, ecg_notnan)

      self.ecg = interpolator.interp_fn(self.t)

    if np.any(np.isnan(self.abp)):

      abp_notnan = self.abp[~np.isnan(self.abp)]
      t_notnan = self.t[~np.isnan(self.abp)]

      interpolator.fit(t_notnan, abp_notnan)

      self.abp = interpolator.interp_fn(self.t)

  def filter(self,
             ecg_critical_frequency = np.array([1, 20]), ecg_butter_type = 'bandpass', ecg_filter_order = 3,
             abp_critical_frequency = 20, abp_butter_type = 'low', abp_filter_order = 3):

    self.ecg = butter(self.ecg, critical_frequency = ecg_critical_frequency,
                        butter_type = ecg_butter_type, filter_order = ecg_filter_order,
                        sampling_rate = 1/self.dt)
    self.abp = butter(self.abp, critical_frequency = abp_critical_frequency,
                        butter_type = abp_butter_type, filter_order = abp_filter_order,
                        sampling_rate = 1/self.dt)

  def get_beat2beat_features(self,
                        z_ecg_amp_critical = 5,
                        window = 0.5,
                        min_prominence = 0.3,
                        min_interval = 0.6, max_interval = 2, y_interval_critical = 1.5):

    ecg, abp = self.ecg, self.abp
    ecg = ecg/ecg.max()

    ecg_d2 = np.diff(np.pad(ecg, (2, 0), mode='edge'), 2)
    ecg_d3 = np.diff(np.pad(ecg, (3, 0), mode='edge'), 3)

    i_ecg_peaks, p_ecg_peaks  = sc.signal.find_peaks(ecg, prominence = 0)
    p_ecg_peaks = p_ecg_peaks['prominences']

    p_ecg_peaks = p_ecg_peaks[(ecg_d2[i_ecg_peaks] < 0)]
    i_ecg_peaks = i_ecg_peaks[ecg_d2[i_ecg_peaks] < 0]

    i_ecg_peaks_all = i_ecg_peaks

    i_ecg_peaks = i_ecg_peaks[p_ecg_peaks > min_prominence]
    p_ecg_peaks = p_ecg_peaks[p_ecg_peaks > min_prominence]

    p_data = p_ecg_peaks.reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, n_init = 'auto')
    kmeans.fit(p_data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    cluster_sizes = np.bincount(labels)

    valid_cluster_idx = np.where(cluster_sizes >= 20)[0]

    centers = centers[valid_cluster_idx]

    i_ecg_peaks = i_ecg_peaks[np.isin(labels, valid_cluster_idx)]
    p_ecg_peaks = p_ecg_peaks[np.isin(labels, valid_cluster_idx)]

    p_data = p_data[np.isin(labels, valid_cluster_idx)]
    labels = labels[np.isin(labels, valid_cluster_idx)]

    max_center_idx = np.argmax(np.sum(centers, axis=1))
    max_center = centers[max_center_idx]
    max_center_data = p_data[labels == max_center_idx]

    sdev = max_center_data.std()
    threshold = max_center - 4*sdev

    # plt.close()
    # fig, ax = plt.subplots(3, 1, figsize = (20, 10))
    # xlim = [None, None] # [0, 10/self.dt]
    # ax[0].plot(np.arange(len(ecg)), ecg, '-')
    # ax[0].plot(i_ecg_peaks, ecg[i_ecg_peaks], '.g', label = 'ecg peaks')
    # ax[0].legend()
    # ax[0].set_xlim(xlim)

    # ax[1].plot(i_ecg_peaks, p_ecg_peaks, '*g', label = 'ecg peak prominence')
    # ax[1].axhline(y=threshold, color='red', linestyle='--')
    # ax[1].legend()

    # ax[2].scatter(p_data, np.zeros_like(p_data), c=labels, cmap='viridis')
    # ax[2].scatter(centers, np.zeros_like(centers), marker='x', color='red', label='Centers')
    # ax[2].axvline(x=threshold, color='red', linestyle='--')
    # ax[2].set_xlabel('Prominences')
    # ax[2].set_title('K-means Clustering of ECG Prominences (K=2)')
    # ax[2].legend()

    # plt.tight_layout()

    i_ecg_peaks = i_ecg_peaks[p_ecg_peaks > threshold] # [labels == max_center_idx] #
    p_ecg_peaks = p_ecg_peaks[p_ecg_peaks > threshold] # [labels == max_center_idx] #

    # plt.close()
    # plt.plot(np.arange(len(ecg)), ecg, '-') ;
    # plt.plot(i_ecg_peaks, ecg[i_ecg_peaks], '.b', label = f'{len(i_ecg_peaks)} ecg peaks') ;
    # # plt.set_xlim([100000, 110000]) ;
    # plt.legend() ;

    i_ecg_r = i_ecg_peaks

    ##
    interval = np.diff(i_ecg_r)*self.dt
    z_interval = (interval - interval.mean())/interval.std()
    y_interval = interval/np.median(interval)
    i_near =np.where((interval < min_interval) | (y_interval < 1/y_interval_critical))[0]

    while i_near.size != 0:

      i_nears = i_near[0] + [0, 1]

      i_discard = i_nears[ecg[i_ecg_r[i_nears]].argmin()]

      i_ecg_r = np.delete(i_ecg_r, i_discard)

      interval = np.diff(i_ecg_r)*self.dt
      z_interval = (interval - interval.mean())/interval.std()
      y_interval = interval/np.median(interval)
      i_near =np.where((interval < min_interval) | (y_interval < 1/y_interval_critical))[0]
    ##

    hr = 60/interval

    # plt.close()
    # fig, ax = plt.subplots(2, 1, figsize = (20, 5)) ;
    # xlim = [15, 25]
    # ax[0].plot(np.arange(len(ecg))*self.dt, ecg) ;
    # ax[0].plot(i_ecg_r*self.dt, ecg[i_ecg_r],'.') ;
    # ax[0].set_xlim(xlim)

    # ax[1].plot(i_ecg_r[1:]*self.dt, interval) ;
    # ax[1].set_xlim(xlim)

    i_ecg_peaks_fill = i_ecg_peaks_all[~np.isin(i_ecg_peaks_all, i_ecg_r)]

    ##
    interval = np.diff(i_ecg_r)*self.dt
    y_interval = interval/np.median(interval)
    i_far = np.where((interval > max_interval) | (y_interval > y_interval_critical))[0]

    k,j = -1,0
    while (i_far.size != 0) & (j>k): #

      k = np.min(i_far)

      for i in range(len(i_far)):

        j_far = i_far[i] + [0, 1]

        j_ecg_peaks = i_ecg_peaks_fill[(i_ecg_peaks_fill > i_ecg_r[j_far[0]]) & (i_ecg_peaks_fill < i_ecg_r[j_far[1]])]
        if len(j_ecg_peaks) > 0:
          i_ecg_r_fill = j_ecg_peaks[ecg[j_ecg_peaks].argmax()]

          # i_ecg_r = np.unique(np.concatenate(i_ecg_r, i_ecg_r_fill))
          idx = np.searchsorted(i_ecg_r, i_ecg_r_fill)
          i_ecg_r = np.unique(np.insert(i_ecg_r, idx, i_ecg_r_fill))

          i_ecg_peaks_fill = i_ecg_peaks_fill[~np.isin(i_ecg_peaks_fill, i_ecg_r)]

        # plt.close()
        # fig, ax = plt.subplots(2, 1, figsize = (20, 5)) ;
        # xlim = [15, 25]
        # ax[0].plot(np.arange(len(ecg))*self.dt,ecg) ;
        # ax[0].plot(i_ecg_r*self.dt, ecg[i_ecg_r],'.b') ;
        # # ax[0].plot(i_ecg_r_fill*self.dt, ecg[i_ecg_r_fill],'or') ;
        # ax[0].set_xlim(xlim)

        # ax[1].plot(i_ecg_r[1:]*self.dt, y_interval) ;
        # ax[1].set_xlim(xlim)

      interval = np.diff(i_ecg_r)*self.dt
      y_interval = interval/np.median(interval)
      i_far = np.where((interval > max_interval) | (y_interval > y_interval_critical))[0]

      j = np.min(i_far[0]) if i_far.size != 0 else -1

    hr = 60/interval
    ##

    # plt.close()
    # fig, ax = plt.subplots(2, 1, figsize = (20, 5)) ;
    # xlim = [None, None]
    # ax[0].plot(np.arange(len(ecg))*self.dt,ecg) ;
    # ax[0].plot(i_ecg_r*self.dt, ecg[i_ecg_r],'.') ;
    # ax[0].set_xlim(xlim)

    # ax[1].plot(i_ecg_r[1:]*self.dt, interval) ;
    # ax[1].set_xlim(xlim)

    interval = np.diff(i_ecg_r)*self.dt
    hr = 60/interval
    self.i_ecg_r = i_ecg_r
    self.interval, self.hr = interval, hr
    self.beat_dt = self.interval.mean().round(2)
    self.beat_t = self.interval.cumsum()

    ##
    i_abp_peaks, i_abp_troughs = sc.signal.find_peaks(abp)[0], sc.signal.find_peaks(-abp)[0]

    i_sbp, i_dbp, mabp = [], [], []
    i = 1
    updated_i_ecg_r = [i_ecg_r[0]]
    while i < len(i_ecg_r):
      j_abp_troughs = i_abp_troughs[(i_abp_troughs > updated_i_ecg_r[-1]) & (i_abp_troughs <= i_ecg_r[i])]
      j_abp_peaks = i_abp_peaks[(i_abp_peaks > updated_i_ecg_r[-1]) & (i_abp_peaks <= i_ecg_r[i])]

      if (len(j_abp_troughs) == 0) or (len(j_abp_peaks) == 0):
        i_ecg_r_min = i_ecg_r[(i-1):i][ecg[i_ecg_r[(i-1):i]].argmin()]
        i_ecg_r = np.delete(i_ecg_r, np.where(i_ecg_r == i_ecg_r_min))

        interval = np.diff(i_ecg_r)*self.dt
        hr = 60/interval
        self.i_ecg_r = i_ecg_r
        self.interval, self.hr = interval, hr
        self.beat_dt = self.interval.mean().round(2)
        self.beat_t = self.interval.cumsum()

        # plt.close()
        # fig, ax = plt.subplots(2,1)
        # ax[0].plot(ecg)
        # ax[0].plot(i_ecg_r, ecg[i_ecg_r], '.')
        # ax[0].set_xlim([i_ecg_r[i]-30/self.dt, i_ecg_r[i]+30/self.dt])

        # ax[1].plot(abp)
        # ax[1].plot(i_dbp,abp[i_dbp],'.')
        # ax[1].plot(i_sbp,abp[i_sbp],'.')
        # ax[1].set_xlim([i_ecg_r[i]-30/self.dt, i_ecg_r[i]+30/self.dt])

      else:
        updated_i_ecg_r.append(i_ecg_r[i])

        i_dbp.append(j_abp_troughs[abp[j_abp_troughs].argmin()])
        i_sbp.append(j_abp_peaks[abp[j_abp_peaks].argmax()])

        mabp.append(abp[i_ecg_r[i-1]:(i_ecg_r[i]+1)].mean())

        i += 1

    i_ecg_r = np.array(updated_i_ecg_r)
    i_sbp, i_dbp = np.array(i_sbp), np.array(i_dbp)
    ##

    # plt.close()
    # fig, ax = plt.subplots(2, 1)
    # xlim = [None, None] # [0, 5/self.dt]
    # ax[0].plot(ecg)
    # ax[0].plot(i_ecg_r, ecg[i_ecg_r], '.g')
    # ax[0].set_xlim(xlim)

    # ax[1].plot(abp)
    # ax[1].plot(i_sbp, abp[i_sbp], '.g')
    # ax[1].plot(i_dbp, abp[i_dbp], '.r')
    # ax[1].set_xlim(xlim)
    #
    # plt.tight_layout()

    sbp, dbp = abp[i_sbp], abp[i_dbp]
    mabp = np.array(mabp)

    # i_sbp, i_dbp = i_sbp[:-1], i_dbp[:-1]
    # i_ecg_r = i_ecg_r[1:]

    self.sbp, self.dbp, self.i_sbp, self.i_dbp, self.mabp = sbp, dbp, i_sbp, i_dbp, mabp

  def generate_beat2beat_variability(self, window_type = 'hann', moving_average_window_len = 120):

    if window_type == 'hann':
      window = sc.signal.windows.hann(moving_average_window_len)
    elif window_type == 'hamming':
      window = sc.signal.windows.hamming(moving_average_window_len)

    self.moving_average_window = sbp_ma = moving_average(self.sbp, window)

    self.sbp_ma, self.dbp_ma = moving_average(self.sbp, window), moving_average(self.dbp, window)
    self.mabp_ma = moving_average(self.mabp, window)

    self.hr_ma = moving_average(self.hr, window)
    self.interval_ma = moving_average(self.interval, window)

    self.sbpv, self.dbpv = self.sbp - self.sbp_ma, self.dbp - self.dbp_ma
    self.mabpv = self.mabp - self.mabp_ma
    self.hrv = self.hr - self.hr_ma
    self.intervalv = self.interval - self.interval_ma

    self.sbpv, self.dbpv, self.mabpv = self.sbpv - self.sbpv.mean(), self.dbpv - self.dbpv.mean(), self.mabpv - self.mabpv.mean()
    self.hrv, self.intervalv = self.hrv - self.hrv.mean(), self.intervalv - self.intervalv.mean()

  def remove_outliers(self, z_abp_change_critical = 4, z_hr_change_critical = 4, interp_type = 'linear'):

    hr, interval, dbp, sbp = self.hr.copy(), self.interval.copy(), self.dbp.copy(), self.sbp.copy()

    ## hr/interval
    i_hr_all = np.arange(len(hr),  dtype = np.compat.long)
    i_hr = i_hr_all

    hr_diff = np.diff(hr)
    z_hr_diff = (hr_diff - hr_diff.mean())/hr_diff.std()

    i_discard = np.where(np.abs(z_hr_diff) > z_hr_change_critical)[0]

    if len(i_discard):
      hr_interpolator = Interpolator(kind = interp_type)
      interval_interpolator = Interpolator(kind = interp_type)
    else:
      hr_interpolator = None
      interval_interpolator = None

    while len(i_discard) > 0:

      j_discard = i_discard[0] + [0, 1]
      j_discard = j_discard[j_discard < len(hr)]

      j_discard = j_discard[np.abs(hr.mean() - hr[j_discard]).argmax()]
      hr = np.delete(hr, j_discard)
      interval = np.delete(interval, j_discard)
      i_hr = np.delete(i_hr, j_discard)

      hr_diff = np.diff(hr)
      z_hr_diff = (hr_diff - hr_diff.mean())/hr_diff.std()

      i_discard = np.where(np.abs(z_hr_diff) > z_hr_change_critical)[0]
    ##

    # plt.figure(1)
    # plt.plot(i_hr_all, self.hr)
    # plt.plot(i_hr, hr)

    ## dbp
    i_dbp_all = np.arange(len(dbp),  dtype = np.compat.long)
    i_dbp = i_dbp_all

    dbp_diff = np.diff(dbp)
    z_dbp_diff = (dbp_diff - dbp_diff.mean())/dbp_diff.std()

    i_discard = np.where(np.abs(z_dbp_diff) > z_abp_change_critical)[0]

    if len(i_discard):
      dbp_interpolator = Interpolator(kind = interp_type)
    else:
      dbp_interpolator = None

    while len(i_discard) > 0:

      j_discard = i_discard[0] + [0, 1]
      j_discard = j_discard[j_discard < len(dbp)]

      j_discard = j_discard[np.abs(dbp.mean() - dbp[j_discard]).argmax()]
      dbp = np.delete(dbp, j_discard)
      i_dbp = np.delete(i_dbp, j_discard)

      dbp_diff = np.diff(dbp)
      z_dbp_diff = (dbp_diff - dbp_diff.mean())/dbp_diff.std()

      i_discard = np.where(np.abs(z_dbp_diff) > z_abp_change_critical)[0]

    # plt.figure(2)
    # plt.plot(i_dbp_all, self.dbp)
    # plt.plot(i_dbp, dbp)
    ##

    ## sbp
    i_sbp_all = np.arange(len(sbp),  dtype = np.compat.long)
    i_sbp = i_sbp_all

    sbp_diff = np.diff(sbp)
    z_sbp_diff = (sbp_diff - sbp_diff.mean())/sbp_diff.std()

    i_discard = np.where(np.abs(z_sbp_diff) > z_abp_change_critical)[0]

    if len(i_discard):
      sbp_interpolator = Interpolator(kind = interp_type)
    else:
      sbp_interpolator = None

    while len(i_discard) > 0:

      j_discard = i_discard[0] + [0, 1]
      j_discard = j_discard[j_discard < len(sbp)]

      j_discard = j_discard[np.abs(sbp.mean() - sbp[j_discard]).argmax()]
      sbp = np.delete(sbp, j_discard)
      i_sbp = np.delete(i_sbp, j_discard)

      sbp_diff = np.diff(sbp)
      z_sbp_diff = (sbp_diff - sbp_diff.mean())/sbp_diff.std()

      i_discard = np.where(np.abs(z_sbp_diff) > z_abp_change_critical)[0]

    # plt.figure(3)
    # plt.plot(i_sbp_all, self.sbp)
    # plt.plot(i_sbp, sbp)
    ##

    ##
    i_min = np.max([np.min(i_hr), np.min(i_dbp), np.min(i_sbp)])
    i_max = np.min([np.max(i_hr), np.max(i_dbp), np.max(i_sbp)])

    i_all = np.arange(i_min, i_max+1, dtype = np.compat.long)
    self.beat_t = self.beat_t[i_all]

    if hr_interpolator is not None:
      hr_interpolator.fit(i_hr, hr)
      interval_interpolator.fit(i_hr, interval)
      self.hr = hr_interpolator.interp_fn(i_all)
      self.interval = interval_interpolator.interp_fn(i_all)
    else:
      self.hr = hr[i_all]
      self.interval = interval[i_all]

    if dbp_interpolator is not None:
      dbp_interpolator.fit(i_dbp, dbp)
      self.dbp = dbp_interpolator.interp_fn(i_all)
    else:
      self.dbp = dbp[i_all]

    if sbp_interpolator is not None:
      sbp_interpolator.fit(i_sbp, sbp)
      self.sbp = sbp_interpolator.interp_fn(i_all)
    else:
      self.sbp = sbp[i_all]

    self.mabp = self.mabp[i_all]
    ##

  def generate_periodogram(self, window_type = 'hann'):
    self.f_psd, self.sbp_psd = periodogram(self.sbpv, fs = 1./self.beat_dt, window = window_type)
    _, self.dbp_psd = periodogram(self.dbpv, fs =  1./self.beat_dt, window = window_type)
    _, self.interval_psd = periodogram(self.intervalv, fs = 1./self.beat_dt, window = window_type)

  def plot_realtime(self, fig_num = 1, zoom_window = [0, 120]):

    fig, ax = plt.subplots(2, 2, figsize=(20,10), num = fig_num)

    ax[0,0].plot(self.t, self.ecg, label = 'ECG')
    ax[0,0].plot(self.t[self.i_ecg_r], self.ecg[self.i_ecg_r], '.g', label = 'R-peak')
    ax[0,0].legend()

    ax[1,0].plot(self.t, self.abp, label = 'ABP')
    ax[1,0].plot(self.t[self.i_dbp], self.abp[self.i_dbp], '.r', label = 'DBP')
    ax[1,0].plot(self.t[self.i_sbp], self.abp[self.i_sbp], '.g', label = 'SBP')
    ax[1,0].legend()

    ax[0,1].plot(self.t, self.ecg, label = 'ECG')
    ax[0,1].plot(self.t[self.i_ecg_r], self.ecg[self.i_ecg_r], '.g', label = 'R-peak')
    ax[0,1].legend()
    ax[0,1].set_xlim(zoom_window)

    ax[1,1].plot(self.t, self.abp, label = 'ABP')
    ax[1,1].plot(self.t[self.i_dbp], self.abp[self.i_dbp], '.r', label = 'DBP')
    ax[1,1].plot(self.t[self.i_sbp], self.abp[self.i_sbp], '.g', label = 'SBP')
    ax[1,1].legend()
    ax[1,1].set_xlim(zoom_window)

    fig.tight_layout()

  def plot_beat2beat(self, fig_num = 1, flim = [0, None]):

    fig, ax = plt.subplots(2,3, figsize=(20,10), num = fig_num)
    ax[0,0].plot(self.beat_t, self.dbp, 'r', alpha = 0.5) ; ax[0,0].set_ylabel('SBP & DBP') ; ax[0,0].set_title('Before MA Removal')
    ax[0,0].plot(self.beat_t, self.dbp_ma, 'r') ; ax[0,1].set_title('120-Beat Moving Average (Hann)')
    ax[0,0].plot(self.beat_t, self.sbp, 'g', alpha = 0.5) ;
    ax[0,0].plot(self.beat_t, self.sbp_ma, 'g') ;

    ax[0,1].plot(self.beat_t, self.sbpv, 'g') ; ax[0,1].set_title('After MA Removal')
    ax[0,1].plot(self.beat_t, self.dbpv, 'r') ;

    ax[0,2].plot(self.f_psd, self.dbp_psd, 'r') ; ax[0,2].set_title('Power Spectrum')
    ax[0,2].plot(self.f_psd, self.sbp_psd, 'g') ;
    ax[0,2].set_xlim(flim)

    ax[1,0].plot(self.beat_t, self.interval, 'b', alpha = 0.5) ; ax[1,0].set_ylabel('I')
    ax[1,0].plot(self.beat_t, self.interval_ma, 'b')

    ax[1,1].plot(self.beat_t, self.intervalv, 'b')

    ax[1,2].plot(self.f_psd, self.interval_psd, 'b')
    ax[1,2].set_xlim(flim)

    fig.tight_layout()
