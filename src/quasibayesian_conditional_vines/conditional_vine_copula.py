import numpy as np
import pyvinecopulib as pv
import torch
import math

from scipy.stats import norm, multivariate_normal
from itertools import chain


class conditional_vine_copula:
    '''
    Class to mimic some of pv.Vinecop for cleaner code.

    Can calculate the pdf of a conditional vine copula when passed a pv.Vinecop object constructed with a conditional set.

    Currently assumes all discrete variables are at the end of the array passed to the vine.
    i.e. n continuous, q discrete, q discrete left limits
    '''
    def __init__(self, conditioning_set, vine:pv.Vinecop = None, n_discrete = 0):
        if len(conditioning_set) == 1:
            print("WARNING: This wil not produce the true conditional pdf, you must not multiply by the marginal of the conditioning variable")
        if len(conditioning_set) == 2:
            print("WARNING: If the copula of the conditioning is not contained in the Vine, this will not give the correct value.")

        if vine is not None:
            self.vine = vine
            self.trees_np = np.fromiter(chain.from_iterable(vine.get_trees()), dtype=object)

            self.subtree_mask = self.__find_subtree_mask(conditioning_set)
            self.subtrees_np = self.trees_np[self.subtree_mask]
            self.ed_set = np.setdiff1d(vine.order, conditioning_set)

        self.conditioning_set = np.array(conditioning_set)
        self.n_discrete = n_discrete
        self.var_types = ["c"]*max(conditioning_set)
        if n_discrete >0:
            self.var_types = ["c"]*(max(conditioning_set)-1)+["d"]*n_discrete
            print(f"var_types = {self.var_types}")


    def fit_from_data(self, data, controls = pv.FitControlsVinecop(), check_vine = False):
        controls.conditioning_set = self.conditioning_set
        self.vine = pv.Vinecop.from_data(data, var_types=self.var_types, controls=controls)

        self.ed_set = np.setdiff1d(self.vine.order, self.conditioning_set)
        self.trees_np = np.fromiter(chain.from_iterable(self.vine.get_trees()), dtype=object)

        self.subtree_mask = self.__find_subtree_mask(self.conditioning_set)
        self.subtrees_np = self.trees_np[self.subtree_mask]

        self.cond_order = np.array(self.vine.order)[-len(self.conditioning_set):]-len(self.ed_set)-1

        if check_vine:
            print(self.vine)

    def __find_subtree_mask(self, conditioning_set:tuple)->np.array:
        '''
        Finds the complement of the sub tree in the list of trees given. Returns a boolean array with True corresponding to the copulas which 
        should be evaluated.

        By definition in the conditioning-set control knob, there wil be a sub-tree containing the conditioning set, fitted first.
        Hence we only need to look for a tree of smaller depth!

        There are specific cases that do not require this function, such as if there are only two conditioned variables, positioned at either end of the order.
        '''
        mask = np.zeros_like(self.trees_np, dtype=bool)

        for i, cop in enumerate(self.trees_np):
            allc = cop['conditioned']+tuple(cop['conditioning'])
            # Is there a number that isn't in the conditioning set? Set to True
            mask[i] = False in np.isin(allc, conditioning_set)
        return mask
    
    @staticmethod
    def __t_idx(t:int, dim:int)->int:
        '''
        Finds the starting index of a tree along a flattened array of copulas.
        '''
        return int(t*(dim+1-(t+1)/2))

    def __eval_h_functions(self, u: np.array)-> tuple: 
        '''
        Evaluates all h functions required for the vine structure, therefore allowing conditional copulas to be evaluated
        from these values (From a structure containing the tree subset to condition on).
        '''
        cops = self.trees_np

        n = u.shape[0]
        d = u.shape[1]
        n_cs = len(cops)
        h_evals = np.zeros((d+n_cs,2,n)) #Flatten over trees and edges so we have
                                                    # a nice data structure to deal with (i.e. numpy/torch)
                                                    # n_cops, 2, n
        h_evals[:d, 0] = h_evals[:d, 1] = u.T
        h_pointers = np.zeros((n_cs, 2, 2), dtype=int)
        end_first_tree = self.__t_idx(1, d-1)
        # Go through the first tree, this works!
        for j, cop in enumerate(cops[0:end_first_tree]):
            c1, c2 = cops[j]['conditioned']
            c1 -= 1
            c2 -= 1
            h_evals[j+d,0] = cop['pair_copula'].hfunc1(u[:, [c1,c2]]) # c2|c1
            h_evals[j+d,1] = cop['pair_copula'].hfunc2(u[:, [c1,c2]]) # c1|c2

            mask = ((c1, c2),(0,0))
            h_pointers[j] = mask
        t=1
        #Go through trees > 1. This does not work with indexing!
        for j, cop in enumerate(cops[end_first_tree:]):
            if end_first_tree+j>=self.__t_idx(t+1, d-1):
                t+=1
            point_start = self.__t_idx(t-1, d-1)
            new_point_start = self.__t_idx(t, d-1)
            old_eval_start = self.__t_idx(t, d)
            #I want to find the h_eval required to compute the next one correctly according to the vine.
            c1, c2 = cop['conditioned']
            cing = cop['conditioning']
            up1 = np.append(cing, c1)
            up2 = np.append(cing, c2)
            nup = t+1
            #print(point_start)
            count = 0

            #print(cop)
            #Find the h_evals in the vine above corresponding to the current copula.
            for k, upcop in enumerate(cops[point_start:new_point_start]):
                allc = upcop['conditioned']+tuple(upcop['conditioning'])

                if np.sum(np.isin(up1, allc))==nup: # Use np.all?
                    idx1 = k
                    count += 1
                    if count == 2:
                        break
                elif np.sum(np.isin(up2, allc))==nup:
                    idx2 = k
                    count += 1
                    if count == 2:
                        break
            #Which h-function should I use?
            if cops[point_start+idx1]['conditioned'][0]==c1:
                funcidx1 = 1
            else:
                funcidx1 = 0
            if cops[point_start+idx2]['conditioned'][0]==c2:
                funcidx2 = 1
            else:
                funcidx2 = 0

            mask = ((old_eval_start+ idx1, old_eval_start+ idx2),(funcidx1,funcidx2))# Points to correct h_evals
            h_pointers[end_first_tree+j] = mask

            #Assign h function evaluations # if statement has not fixed it.
            h_evals[d+end_first_tree+j,0] = cop['pair_copula'].hfunc1(h_evals[mask].T)
            h_evals[d+end_first_tree+j,1] = cop['pair_copula'].hfunc2(h_evals[mask].T)

        return h_evals, (h_pointers[:,0], h_pointers[:,1])

    def pdf(self, data: np.array)->np.array:
        density = 1 # This expands out to nd automatically if data is nd
        if self.n_discrete>0:
            #could speed this up by working with the original structure given.
            edge_pdfs = np.fromiter(chain.from_iterable(self.vine.pdf_full(data)['pdf_edges']), dtype=object)
            for i, mask in enumerate(self.subtree_mask):
                if mask:
                    density *= edge_pdfs[i]

        else:
            
            h_evals, h_points = self.__eval_h_functions(data)

            for i, mask in enumerate(self.subtree_mask):
                if mask:
                    density *= self.trees_np[i]['pair_copula'].pdf(h_evals[h_points][i].T)

        return density

    @staticmethod
    def __get_filled_hfunc(hfuncs):
        for hfunc in hfuncs:
            if hfunc is not None and hfunc.size:
                return hfunc
        print("No filled h-function was found.")

        return None

    def __resolve_independence_hfunc(self, pdf_full, data, tree_idx, edge_idx,
                                     hfunc_idx, sub=False, resolving=None):
        """Recover a missing h-function for an independence edge."""
        if resolving is None:
            resolving = set()

        key = (tree_idx, edge_idx, hfunc_idx, sub)
        if key in resolving:
            return None
        resolving.add(key)

        field = ("hfunc1_sub" if sub else "hfunc1") if hfunc_idx == 0 else (
            "hfunc2_sub" if sub else "hfunc2"
        )
        stored = pdf_full[field][tree_idx][edge_idx]
        if stored.size:
            return stored

        edge = self.vine.get_trees()[tree_idx][edge_idx]
        if edge["pair_copula"].family != pv.BicopFamily.indep:
            return None

        conditioned = edge["conditioned"]
        variable = conditioned[1] if hfunc_idx == 0 else conditioned[0]

        if tree_idx == 0:
            dimension = self.vine.dim
            variable_idx = variable - 1
            if sub and variable > dimension - self.n_discrete:
                variable_idx = dimension + variable - (dimension - self.n_discrete) - 1
            return data[:, variable_idx]

        conditioning = set(edge["conditioning"])
        required = conditioning | {variable}
        lower_edges = self.vine.get_trees()[tree_idx - 1]

        for lower_idx, lower_edge in enumerate(lower_edges):
            lower_set = set(lower_edge["conditioned"]) | set(lower_edge["conditioning"])
            if lower_set != required:
                continue

            lower_conditioned = lower_edge["conditioned"]
            if variable == lower_conditioned[0]:
                lower_hfunc_idx = 1
            elif variable == lower_conditioned[1]:
                lower_hfunc_idx = 0
            else:
                continue

            return self.__resolve_independence_hfunc(
                pdf_full,
                data,
                tree_idx - 1,
                lower_idx,
                lower_hfunc_idx,
                sub=sub,
                resolving=resolving,
            )

        return None

    def __get_tree_hfuncs(self, pdf_full, data, tree_idx, hfunc_idx, sub=False):
        """Return stored h-functions, filling independence edges when possible."""
        values = []
        for edge_idx in range(len(self.vine.get_trees()[tree_idx])):
            values.append(self.__resolve_independence_hfunc(
                pdf_full,
                data,
                tree_idx,
                edge_idx,
                hfunc_idx,
                sub=sub,
            ))
        return values

    def __get_final_hfunc_input(self, pdf_full, data, tree_idx, variable, sub=False):
        """Get the penultimate h-function corresponding to a final-edge endpoint."""
        for edge_idx, edge in enumerate(self.vine.get_trees()[tree_idx]):
            edge_variables = set(edge["conditioned"]) | set(edge["conditioning"])
            if variable not in edge_variables:
                continue

            if edge["conditioned"][0] == variable:
                hfunc_idx = 1
            elif edge["conditioned"][1] == variable:
                hfunc_idx = 0
            else:
                continue

            hfunc = self.__resolve_independence_hfunc(
                pdf_full, data, tree_idx, edge_idx, hfunc_idx, sub=sub
            )
            if hfunc is not None and hfunc.size:
                return hfunc

        return None

    def cdf_implicit(self, data):
        '''
        Currently implementing the implicit conditional copula in one dimension. This is implicit as we can simply evaluate
        the last h function of the tree structure as the h functions are conditional cdfs by construction.
        '''
        if len(self.ed_set) == 1:
            if self.n_discrete>0:
                pdf_full = self.vine.pdf_full(data)

                penultimate = len(self.vine.get_trees()) - 2

                final_edge = self.trees_np[-1]
                final_conditioned = final_edge["conditioned"]
                u1 = self.__get_final_hfunc_input(
                    pdf_full, data, penultimate, final_conditioned[0]
                )
                u2 = self.__get_final_hfunc_input(
                    pdf_full, data, penultimate, final_conditioned[1]
                )

                u1_sub = self.__get_final_hfunc_input(
                    pdf_full, data, penultimate, final_conditioned[0], sub=True
                )
                u2_sub = self.__get_final_hfunc_input(
                    pdf_full, data, penultimate, final_conditioned[1], sub=True
                )

                if (u1 is None) or (u2 is None) :
                    if u1 is None:
                        print("u1")
                    if u2 is None:
                        print("u2")
                    print(pdf_full)
                

                if u1_sub is None:
                    u1_sub = u1
                if u2_sub is None:
                    u2_sub = u2
                final_inputs = np.column_stack((u1, u2, u1_sub, u2_sub))

                if final_conditioned[0] == self.ed_set:
                    return final_edge["pair_copula"].hfunc2(final_inputs)
                elif final_conditioned[1] == self.ed_set:
                    return final_edge["pair_copula"].hfunc1(final_inputs)
                else:
                    raise ValueError("The conditioned set implied from the given conditioning set and the vine copula"
                    " is not in the last copula. The vine is not set up correctly.")


            else:
                h_evals, _ = self.__eval_h_functions(data)

                if self.trees_np[-1]["conditioned"][0] == self.ed_set:
                    return h_evals[-1][1]
                elif self.trees_np[-1]["conditioned"][1] == self.ed_set:
                    return h_evals[-1][0]
                else:
                    raise ValueError("The conditioned set implied from the given conditioning set and the vine copula"
                    " is not in the last copula. The vine is not set up correctly.")

        else:
            raise NotImplementedError("Not implemented as 2 dimensions requires further structure, and >2D there is no currently known implicit form")

    def cdf_numerical(self, data, n_points = 6):
        '''
        Numerical integration over all conditioned variables. This is more complex than it seems and infeasable in high dimensions,
        try Monte-Carlo Methods using Vinecop.simulate_conditional.
        '''
        
        ed_idxs = self.ed_set-1
        ing_idxs = self.conditioning_set-1

        if len(ing_idxs) == 0:
            print("try cdf_mc")
            raise NotImplementedError
        elif len(ed_idxs) == 1:
            data_ing = data[:, ing_idxs]
            ed_idx = ed_idxs[0]
            data_ed = data[:, ed_idx]

            zero_to_data = np.vstack([
                np.linspace(1e-10, value, n_points)
                for value in data_ed
            ])
            print("zero to data :",zero_to_data)
            numer_inputs = np.repeat(data[:, None, :], n_points, axis=1)
            print(numer_inputs)
            numer_inputs[:, :, ed_idx] = zero_to_data
            print(numer_inputs)
            numer_inputs = numer_inputs.reshape(-1, data.shape[1])


            print(numer_inputs)
            y = self.pdf(numer_inputs).reshape(data.shape[0], n_points)
            print(y)

            return np.trapezoid(y, x=zero_to_data, axis=1)

        else:
            print("Not yet implemented, try cdf_mc")
            raise NotImplementedError
 
    def cdf_mc(self, data, n_samples = 100000):
        #Need to change
        if len(self.conditioning_set)>0:
            print(self.conditioning_set)
            data_ing = data[:, self.conditioning_set-1]
            data_ed = data[:, self.ed_set-1]
            conds = np.zeros((data_ing.shape[0]*n_samples, data_ing.shape[1]))
            # How to parallelise/optimise this?
            for i, point in enumerate(data_ing):
                conds[i*n_samples: (i+1)*n_samples] = np.repeat([point], n_samples, axis=0)

            samples = self.vine.simulate_conditional(conds, True, num_threads=4)
            samples = samples.reshape((data.shape[0], n_samples, data.shape[1]))

            p = np.mean(np.all(np.less(samples[:, :, self.ed_set-1], data_ed[:, None, :]), axis=2), axis = 1)
            return p
        else:
            samples = self.vine.simulate(n_samples)
            p = np.mean(np.all(np.less(samples, data[:, None, :]), axis=2), axis = 1)
            return p

    def inverse_transform():
        pass

    def conditional_sample(self, cond_data, n_samples=100):
        #assert cond_data.shape == self.conditioning_set.shape, f"The conditioning data must have the same shape as the conditioning set of the vine. cond_data.shape = {cond_data.shape}, conditioning_set.shape = {self.conditioning_set.shape}"
        u_cond = np.repeat(cond_data, repeats=n_samples, axis=0)

        u_cond = u_cond[:, self.cond_order]
        if np.any(u_cond==0):
            print("0 value sampled")
        elif np.any(u_cond == 1):
            print("1 value sampled")
        
        u_cond = np.clip(u_cond, 1e-8, 1 - 1e-8)

        return self.vine.simulate_conditional(u_cond)