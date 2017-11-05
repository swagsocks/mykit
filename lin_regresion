import numpy as np
import pandas as pd

#X_frame and y_frame are Pandas DataFrames

def beta_hat(X_frame, y_frame):
	X = X_frame
	X['1'] = 1
	y = y_frame
	betas = np.dot(np.dot(np.linalg.inv(np.dot(np.array(np.transpose(X_frame)), np.array(X_frame))), np.array(np.transpose(X_frame))), np.array(y_frame))
	return betas
	
betas = beta_hat(X_frame, y_frame)
	
def st_error(X_frame, y_frame, betas):
	residual = y_frame- np.dot(X_frame, betas)
	o = (np.dot(np.transpose(residual), residual)/X_frame.shape[0]) **.5
	return o
	
def R2(X_frame, y_frame, betas):
	wise = np.dot(np.transpose(y_frame), y_frame)
	wise_hat = (np.dot(np.dot(np.transpose(betas), np.transpose(X_frame)), y_frame))
	SS_res = wise -wise_hat
	y_hat = np.dot(X_frame, betas)
	SS_tot = sum(np.subtract(y_hat, float(np.mean(y_frame)))**2)
	SS_T = wise - float((np.sum(y_frame)**2)/X_frame.shape[0])
	R2 = 1- (SS_res/SS_tot)
	F = SS_tot/(X_frame.shape[1]-1)/(SS_res/(X_frame.shape[0]-X_frame.shape[1]))
	R2_adj = 1- (SS_res/(X_frame.shape[0]-X_frame.shape[1])/(SS_tot/(X_frame.shape[0]-1)))
	return F, R2, R2_adj
	
def Significance(X_frame, y_frame, betas, index):
	X_frame1 = X_frame
	X_frame1['1'] = 1
	cols = X_frame1.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	X_frame1 = X_frame1[cols] 
	o2 = (st_error(X_frame, y_frame, betas))**2
	X_thang = np.linalg.inv(np.dot(np.transpose(X_frame1), X_frame1))
	new = np.diag(o2 * X_thang)
	se_B = new[index]**.5
	t = beta[index]/se_B
	if X_frame.shape[0] > 120:
		if t > 3.373:
			return t, '.0005'
		elif t > 3.160:
			return t, '.001'
		elif t > 2.617:
			return t, '.005'
		elif t > 2.358:
			return t, '.01'
		elif t >  1.980:
			return t, '.025'
		elif t >  1.658:
			return t, '.05'
		elif t >  1.289:
			return t, '.1'
		else:
			return t, 'insignificant'
	else: 
		return t, 'insignificant'
