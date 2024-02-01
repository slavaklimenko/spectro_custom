import pickle
import numpy as np
import emcee
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from scipy.stats import gaussian_kde
from spectro.stats import distr1d
from spectro.a_unc import a

if 1:
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.unicode'] = True
    #matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['axes.titlesize'] = 10

def readChain(filename="output/mcmc.hdf5",parsname='output/pars.pkl'):
    with open(parsname, "rb") as f:
        pars = pickle.load(f)

    backend = emcee.backends.HDFBackend(filename)
    lnprobs = backend.get_log_prob()
    samples = backend.get_chain()
    return pars, samples, lnprobs

def stats(t='cols',burnin=10,filename="output/mcmc.hdf5",addconstrain=False,parsname='output/pars.pkl'):
    pars, samples, lnprobs = readChain(filename=filename,parsname=parsname)
    nsteps, nwalkers, = lnprobs.shape
    inds = np.where(lnprobs == np.max(lnprobs))

    if addconstrain:
        indx = pars.index('N_1_HI')
        mask = samples[burnin:, :, indx] < 15.5
    else:
        indx = pars.index('N_1_SiIII')
        mask = samples[burnin:, :, indx] > 0
    if t == 'fit':
        names = pars
        k = int(np.size(pars))  # samples.shape[1]
        n_hor = int(k ** 0.5)
        n_hor = np.max([n_hor, 2])
        n_vert = k // n_hor + 1 if k % n_hor > 0 else k // n_hor
        n_vert = np.max([n_vert, 2])

        fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
        k = 0
        for i, p in enumerate(pars):
            print(i, p)
            if p in names:
                s = (samples[burnin:, :, i])[mask].flatten()
                x = np.linspace(np.min(s), np.max(s), 100)
                kde = gaussian_kde(s)
                d = distr1d(x, kde(x))
                d.dopoint()
                if 1:
                    print('3-sigma estimate')
                    d.dointerval(conf=0.997)
                    # res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0],
                    #        self.parent.fit.getPar(p).form)
                    # d.plot()
                    # print(res.val,res.plus, res.minus)
                    sigplus = str(d.interval[1])
                d.dointerval()
                if 'z' in p or 'b' in p:
                    atype = 'dec'
                else:
                    atype = 'log'
                res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0],f=atype)
                print(res.plus, res.minus)
                f = np.asarray([res.plus, res.minus])
                f = int(np.round(np.abs(np.log10(np.min(f[np.nonzero(f)])))) + 2)
                print(p, res.latex(f=f))
                vert, hor = k // n_hor, k % n_hor
                k += 1
                d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                ax[vert, hor].yaxis.set_ticklabels([])
                ax[vert, hor].yaxis.set_ticks([])
                ax[vert, hor].text(.05, .9, str(p).replace('_', ' '), ha='left', va='top',
                                   transform=ax[vert, hor].transAxes)
                ax[vert, hor].text(.95, .9, res.latex(f=f,base=0),
                                   ha='right', va='top', transform=ax[vert, hor].transAxes)
                ax[vert, hor].text(.7, .7, '3sig: ' + sigplus, transform=ax[vert, hor].transAxes)
                # ax[vert, hor].set_title(pars[i].replace('_', ' '))
            if 'z' in p:
                zgal = 0.06013
                print('vel shift:',((res+1)/(1+zgal)-1)*299792)
    else:

        #sampler = (samples[burnin:, :, :])[mask].flatten()
        ndim = samples.shape[2]
        chain = (samples[burnin:,:, :])[mask].reshape((-1, ndim))

        if t == 'cols':
            # mask = np.array([p.show for p in self.parent.fit.list_fit()])
            names = []
            for p in pars:
                if 'N' in p:
                    names.append(str(p).split('_')[-1])
            print('names', names)
            sp = set()
            for s in names:
                if s not in sp:
                    sp.add(s+'_total')
            print('sp', sp)

            n_hor = int(len(sp) ** 0.5)
            if n_hor <= 1:
                n_hor = 2
            n_vert = len(sp) // n_hor + 1 if len(sp) % n_hor > 0 else len(sp) // n_hor
            if n_vert <= 1:
                n_vert = 2
            fig, ax = plt.subplots(nrows=n_vert, ncols=n_hor, figsize=(6 * n_vert, 4 * n_hor))
            i = 0
            for k in sp:
                if 'total' in k:
                    inds = np.where([k.split('_')[0] in str(s) and str(s)[0] == 'N' for s in pars])[0]
                if 1:
                    print('inds', inds)
                    if len(inds) > 1:
                        d = distr1d(np.log10(np.sum(10 ** chain[:, inds], axis=1)))
                        d.dopoint()
                        d.dointerval()
                        res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0])

                        vert, hor = i // n_hor, i % n_hor
                        i += 1
                        d.plot(conf=0.683, ax=ax[vert, hor], ylabel='')
                        ax[vert, hor].yaxis.set_ticklabels([])
                        ax[vert, hor].yaxis.set_ticks([])
                        ax[vert, hor].text(.1, .9, k.replace('_', ' '), ha='left', va='top',
                                           transform=ax[vert, hor].transAxes)
                        ax[vert, hor].text(.95, .9, res,
                                           ha='right', va='top', transform=ax[vert, hor].transAxes)

    if i < n_hor * n_vert:
        for i in range(i + 1, n_hor * n_vert):
            vert, hor = i // n_hor, i % n_hor
            fig.delaxes(ax[vert, hor])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.show()

def plot_graph(burnin=10,filename="output/mcmc.hdf5",addconstrain=False,parsname='output/pars.pkl',multisolution = 0):
    pars, samples, lnprobs = readChain(filename=filename,parsname=parsname)
    nsteps, nwalkers, = lnprobs.shape
    print(pars,nsteps, nwalkers)
    inds = np.where(lnprobs == np.max(lnprobs))

    fig1,ax = plt.subplots(1,4,figsize=(9,2),sharey=True)
    fig1.subplots_adjust(hspace=0.0, wspace=0.1)

    if 1:
        colors = ['tab:blue','tab:green','tab:purple','tab:orange']
        labels = ['Comp 1','Comp 2','Comp 3','Comp 4']
        indx = pars.index('N_0_HI')
        mask = samples[burnin:, :, indx] > 0
        for i, p in enumerate(pars):
            sys = int(p.split('_')[1])
            hor = -1
            if 'N' in p and 'HI' in p:
                hor = 1
            elif 'N' in p and 'SiIII' in p:
                hor = 2
            elif 'b' in p:
                hor = 0
            if hor>=0:

                print(i, p,sys,hor)

                s = (samples[burnin:, :, i])[mask].flatten()
                x = np.linspace(np.min(s), np.max(s), 300)
                kde = gaussian_kde(s)
                conf = 0.683
                if sys == 1 and multisolution:
                    conf = None
                d = distr1d(x, kde(x),mode='max')
                if hor == 0:
                    d.plot(conf=None, ax=ax[hor], ylabel='', color=colors[sys],label=labels[sys])
                else:
                    d.plot(conf=None, ax=ax[hor], ylabel='', color=colors[sys])
                if conf is not None:
                    d = distr1d(x, kde(x))
                    d.dointerval(conf=conf, kind='center')
                    print('interval plot', d.interval)
                    mask2 = np.logical_and(x >= d.interval[0], x <= d.interval[1])
                    d = distr1d(x, kde(x), mode='max')
                    ax[hor].fill_between(x[mask2], d.inter(x)[mask2], facecolor=colors[sys], alpha=0.5,
                                         interpolate=True)
                if sys == 1 and multisolution:
                    print('calc multi pdf')
                    sp_lim = 'N_1_HI'
                    sys_lim = int(sp_lim.split('_')[1])
                    indx = pars.index(sp_lim)

                    # First solution
                    mask_lim = samples[burnin:, :, indx] < 15.8

                    s_add = (samples[burnin:, :, i])[mask_lim].flatten()
                    x_add = np.linspace(np.min(s_add), np.max(s_add), 300)
                    kde_add = gaussian_kde(s_add)
                    conf = 0.683
                    #if sys == 1:
                    #    conf = None
                    #d = distr1d(x, kde(x), mode='max')
                    #d.plot(conf=None, ax=ax[hor], ylabel='', color=colors[sys])
                    #if conf is not None:
                    d_add = distr1d(x_add, kde_add(x_add))
                    d_add.dointerval(conf=conf, kind='center')

                    print('interval plot', d_add.interval)
                    mask2 = np.logical_and(x >= d_add.interval[0], x <= d_add.interval[1])
                    ax[hor].fill_between(x[mask2], d.inter(x)[mask2], facecolor=colors[sys], alpha=0.7, interpolate=True)


                    # Second solution
                    mask_lim = samples[burnin:, :, indx] > 15.8
                    s_add = (samples[burnin:, :, i])[mask_lim].flatten()
                    x_add = np.linspace(np.min(s_add), np.max(s_add), 300)
                    kde_add = gaussian_kde(s_add)
                    conf = 0.683
                    # if sys == 1:
                    #    conf = None
                    # d = distr1d(x, kde(x), mode='max')
                    # d.plot(conf=None, ax=ax[hor], ylabel='', color=colors[sys])
                    # if conf is not None:
                    d_add = distr1d(x_add, kde_add(x_add))
                    d_add.dointerval(conf=conf, kind='center')

                    print('interval plot', d_add.interval)
                    mask2 = np.logical_and(x >= d_add.interval[0], x <= d_add.interval[1])
                    ax[hor].fill_between(x[mask2], d.inter(x)[mask2], facecolor=colors[sys], alpha=0.2, interpolate=True)


        #plt.show()

        if 1:
            hor = 3
            ndim = samples.shape[2]
            chain = (samples[burnin:,:, :])[mask].reshape((-1, ndim))
            names = []
            for p in pars:
                if 'N' in p and 'HI' in p:
                    names.append(str(p).split('_')[-1])
                if 'N' in p and 'SiIII' in p:
                    names.append(str(p).split('_')[-1])
            print('names', names)
            sp = set()
            for s in names:
                if s not in sp:
                    sp.add(s+'_total')
            print('sp', sp)

            color_tot = ['red','black']
            labels_tot= ['HI','SiIII']
            for k in sp:
                if 'total' in k:
                    inds = np.where([k.split('_')[0] in str(s) and str(s)[0] == 'N' for s in pars])[0]
                if 1:
                    print('inds:',k, inds)
                    if len(inds) > 0:
                        #s = np.log10(np.sum(10 ** chain[:, inds]))
                        d = distr1d(np.log10(np.sum(10 ** chain[:, inds], axis=1)), mode='max')
                        #d = distr1d(s, axis=1)
                        #d.dopoint()
                        #d.dointerval()
                        #res = a(d.point, d.interval[1] - d.point, d.point - d.interval[0])
                        #i += 1
                        if 'SiIII' in k:
                            i=1
                        else:
                            i =0
                        print(i,k,labels_tot[i],color_tot[i])
                        d.plot(conf=None, ax=ax[hor], ylabel='',color=color_tot[i],label=labels_tot[i])
                        #plt.show()
                        x = np.linspace(np.min(d.x), np.max(d.x), 300)
                        #ax[hor].yaxis.set_ticklabels([])
                        #ax[hor].yaxis.set_ticks([])
                        #ax[hor].text(.1, .9, k.replace('_', ' '), ha='left', va='top', transform=ax[vert, hor].transAxes)
                        #ax[hor].text(.95, .9, res, ha='right', va='top', transform=ax[vert, hor].transAxes)\
                if multisolution:
                    print('calc multi pdf')
                    mask_add = samples[burnin:, :, indx] < 15.8
                    chain_add = (samples[burnin:, :, :])[mask_add].reshape((-1, ndim))
                    for k in sp:
                        if 'total' in k:
                            inds = np.where([k.split('_')[0] in str(s) and str(s)[0] == 'N' for s in pars])[0]
                        if 1:
                            print('inds', inds)
                            if len(inds) > 1:
                                d_add = distr1d(np.log10(np.sum(10 ** chain_add[:, inds], axis=1)))
                                d_add.dointerval(conf=conf, kind='center')

                                print('interval plot', d_add.interval)

                                if 'SiIII' in k:
                                    i = 1
                                else:
                                    i = 0
                                print(i, k, labels_tot[i], color_tot[i])
                                mask_add = np.logical_and(x >= d_add.interval[0], x <= d_add.interval[1])
                                ax[hor].fill_between(x[mask_add], d.inter(x)[mask_add], facecolor=color_tot[i], alpha=0.7,
                                                     interpolate=True)

                    mask_add = samples[burnin:, :, indx] > 15.8
                    chain_add = (samples[burnin:, :, :])[mask_add].reshape((-1, ndim))
                    for k in sp:
                        if 'total' in k:
                            inds = np.where([k.split('_')[0] in str(s) and str(s)[0] == 'N' for s in pars])[0]
                        if 1:
                            print('inds', inds)
                            if len(inds) > 1:
                                d_add = distr1d(np.log10(np.sum(10 ** chain_add[:, inds], axis=1)))
                                d_add.dointerval(conf=conf, kind='center')

                                print('interval plot', d_add.interval)

                                mask_add = np.logical_and(x >= d_add.interval[0], x <= d_add.interval[1])
                                ax[hor].fill_between(x[mask_add], d.inter(x)[mask_add], facecolor='red', alpha=0.2,
                                                     interpolate=True)
                else:
                    print('calc multi pdf')
                    chain_add = (samples[burnin:, :, :]).reshape((-1, ndim))

                    if 'total' in k:
                        inds = np.where([k.split('_')[0] in str(s) and str(s)[0] == 'N' for s in pars])[0]
                    if 1:
                        print('inds', inds, k)
                        if len(inds) > 0:
                            d_add = distr1d(np.log10(np.sum(10 ** chain_add[:, inds], axis=1)))
                            d_add.dointerval(conf=conf, kind='center')

                            print('interval plot', d_add.interval)

                            mask_add = np.logical_and(x >= d_add.interval[0], x <= d_add.interval[1])
                            ax[hor].fill_between(x[mask_add], d.inter(x)[mask_add], facecolor=color_tot[i], alpha=0.7,
                                                 interpolate=True)




    fontsize = 9
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    #for axs in ax[:]:
    #    axs.yaxis.set_ticklabels([])
    #    axs.yaxis.set_ticks([])
    ax[1].set_xlabel('$\\log N({\\rm HI})$',fontsize=fontsize)
    ax[2].set_xlabel('$\\log N({\\rm SiIII})$', fontsize=fontsize)
    ax[0].set_xlabel('$b, ({\\rm km\\,s^{-1}})$',fontsize=fontsize)
    ax[3].set_xlabel('$\\log N({\\rm X_{\\rm tot}})$',fontsize=fontsize)
    #ax[0].set_ylabel('PDF(J0950+4309)', fontsize=fontsize)
    #ax[0].set_ylabel('PDF(J0758+4219)', fontsize=fontsize)
    #ax[0].set_ylabel('PDF(J1237+4447)', fontsize=fontsize)
    ax[0].set_ylabel('PDF(J2123-0050)', fontsize=fontsize)

    ax[0].set_xlim(15,100)
    #ax[1].set_xlim(12.5,19)
    #ax[2].set_xlim(11.5, 15)
    #ax[3].set_xlim(12.5, 19)
    ax[1].set_xlim(12.5, 20)
    ax[2].set_xlim(11.5,17)
    ax[3].set_xlim(12.5, 20)

    for axs in ax[:]:
        axs.set_ylim(0,1)
        axs.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                        top='True')
        axs.tick_params(which='major', length=8)
        axs.tick_params(which='minor', length=5)
        axs.yaxis.set_minor_locator(AutoMinorLocator(5))
        axs.yaxis.set_major_locator(MultipleLocator(0.5))

    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].xaxis.set_major_locator(MultipleLocator(20))
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].xaxis.set_major_locator(MultipleLocator(2))
    ax[2].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[2].xaxis.set_major_locator(MultipleLocator(1))
    ax[3].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[3].xaxis.set_major_locator(MultipleLocator(2))

    ax[0].legend(fontsize=fontsize,loc='upper right')
    ax[3].legend(fontsize=fontsize, loc='lower left')


    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0)

    if 1:
        fig1.savefig('output/likelihood_J2130.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #filename = '/home/slava/science/research/SouthCarolina/COS/result/j1237/mcmc/mcmc_HI_SiIII.hdf5'
    #parsname = '/home/slava/science/research/SouthCarolina/COS/result/j1237/mcmc/pars_HI_SiIII.pkl'
    #filename = '/home/slava/science/research/SouthCarolina/COS/result/j0758/mcmc/mcmc.hdf5'
    #parsname = '/home/slava/science/research/SouthCarolina/COS/result/j0758/mcmc/pars.pkl'
    #filename = '/home/slava/science/research/SouthCarolina/COS/result/j0950/mcmc/fixed_velocities/2comp/version_2/mcmc.hdf5'
    #parsname = '/home/slava/science/research/SouthCarolina/COS/result/j0950/mcmc/fixed_velocities/2comp/version_2/pars.pkl'
    filename = '/home/slava/science/research/SouthCarolina/COS/result/j2130/mcmc/1comps/mcmc.hdf5'
    parsname = '/home/slava/science/research/SouthCarolina/COS/result/j2130/mcmc/1comps/pars.pkl'
    pars, samples, lnprobs = readChain(filename=filename)
    #stats(filename=filename,burnin=300,parsname = parsname)
    plot_graph(filename=filename,burnin=800,parsname = parsname,addconstrain=True,  multisolution = 0)
    print('The end')