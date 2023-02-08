import dolfin as dl
from .variables import STATE, CONTROL
from .augmentedVector import AugmentedVector


class Penalization:
    def init_vector(self, z):
        raise NotImplementedError("Child class should implement init_vector")

    def cost(self, z):
        raise NotImplementedError("Child class should implement cost")

    def grad(self, z, out):
        raise NotImplementedError("Child class should implement grad")

    def hessian(self, z, zhat, out):
        raise NotImplementedError("Child class should implement hessian")


class MultiPenalization(Penalization):
    def __init__(self, Vh, penalization_list):
        """
        List of penalizations 
        """
        self.Vh = Vh 
        self.helper = dl.Function(Vh[CONTROL]).vector()
        self.penalization_list = penalization_list
    
    def cost(self, z):
        cost = 0
        for penalization in self.penalization_list:
            cost += penalization.cost(z)
        return cost 

    def grad(self, z, out):
        out.zero()
        for penalization in self.penalization_list:
            penalization.grad(z, self.helper)
            out.axpy(1.0, self.helper)

    def hessian(self, z, zhat, out):
        out.zero()
        for penalization in self.penalization_list:
            penalization.hessian(z, zhat, self.helper)
            out.axpy(1.0, self.helper)


class L2Penalization(Penalization):
    def __init__(self, Vh, alpha):
        self.Vh = Vh
        self.alpha = alpha

        z_trial = dl.TrialFunction(self.Vh[CONTROL])
        z_test = dl.TestFunction(self.Vh[CONTROL])

        # Do we need a backend type?
        self.M = dl.assemble(dl.inner(z_trial, z_test) * dl.dx)
        self.Mz = dl.Function(self.Vh[CONTROL]).vector()

    def init_vector(self, v, dim=0):
        if isinstance(v, AugmentedVector):
            pass 
        else:
            self.M.init_vector(v, dim)


    def cost(self, z):
        if isinstance(z, AugmentedVector):
            z_sub = z.get_vector()
            self.M.mult(z_sub, self.Mz)
            return self.alpha * z_sub.inner(self.Mz)
        else:
            self.M.mult(z, self.Mz)
            return self.alpha * z.inner(self.Mz)

    def grad(self, z, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            z_sub = z.get_vector() 
            self.M.mult(z_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(z, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

    def hessian(self, z, zhat, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            zhat_sub = zhat.get_vector() 
            self.M.mult(zhat_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(zhat, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)



class WeightedL2Penalization(Penalization):
    def __init__(self, Vh, M, alpha):
        self.Vh = Vh
        self.M = M 
        self.alpha = alpha
        self.Mz = dl.Function(self.Vh[CONTROL]).vector()

    def init_vector(self, v, dim=0):
        if isinstance(v, AugmentedVector):
            pass 
        else:
            self.M.init_vector(v, dim)


    def cost(self, z):
        if isinstance(z, AugmentedVector):
            z_sub = z.get_vector()
            self.M.mult(z_sub, self.Mz)
            return self.alpha * z_sub.inner(self.Mz)
        else:
            self.M.mult(z, self.Mz)
            return self.alpha * z.inner(self.Mz)

    def grad(self, z, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            z_sub = z.get_vector() 
            self.M.mult(z_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(z, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

    def hessian(self, z, zhat, out):
        out.zero()
        if isinstance(z, AugmentedVector):
            out_sub = out.get_vector()
            zhat_sub = zhat.get_vector() 
            self.M.mult(zhat_sub, self.Mz)
            out_sub.axpy(2.0*self.alpha, self.Mz)
        else:
            self.M.mult(zhat, self.Mz)
            out.axpy(2.0*self.alpha, self.Mz)

