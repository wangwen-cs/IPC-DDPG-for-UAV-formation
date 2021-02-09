from math import cos, sin


class UavModel:

    def __init__(self,
                 mass=0.625,
                 length=0.1275,
                 kf=2.103e-6,
                 km=2.091e-8,
                 ixx=2.3e-3,
                 iyy=2.4e-3,
                 izz=2.6e-3):
        self.m = mass
        self.length = length
        self.kf = kf
        self.km = km
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz


class UavDynamics:

    G = 9.8

    def __init__(self, uav: UavModel):
        self.uav = uav
        self.x = 0
        self.y = 0
        self.z = 0
        self.d_x = 0
        self.d_y = 0
        self.d_z = 0
        self.dd_x = 0
        self.dd_y = 0
        self.dd_z = 0

        self.phi = 0
        self.theta = 0
        self.psi = 0
        self.d_phi = 0
        self.d_theta = 0
        self.d_psi = 0
        self.dd_phi = 0
        self.dd_theta = 0
        self.dd_psi = 0

    def update_with_control(self, omega, dt=0.1):
        f = self.uav.kf * (omega[0] ** 2 + omega[1] ** 2 + omega[2] ** 2 + omega[3] ** 2)
        tau_x = self.uav.kf * self.uav.length * (omega[1] ** 2 - omega[3] ** 2)
        tau_y = self.uav.kf * self.uav.length * (-omega[0] ** 2 + omega[2] ** 2)
        tau_z = self.uav.km * (omega[0] ** 2 - omega[1] ** 2 + omega[2] ** 2 - omega[3] ** 2)

        self.dd_x = (cos(self.phi) * sin(self.theta) * cos(self.psi) + sin(self.phi) * sin(self.psi)) * f / self.uav.m
        self.dd_y = (cos(self.phi) * sin(self.theta) * sin(self.psi) - sin(self.phi) * cos(self.psi)) * f / self.uav.m
        self.dd_z = (cos(self.phi) * cos(self.theta)) * f - UavDynamics.G

        self.d_x += self.dd_x * dt
        self.d_y += self.dd_y * dt
        self.d_z += self.dd_z * dt

        self.x += self.d_x * dt
        self.y += self.d_y * dt
        self.z += self.d_z * dt

        self.dd_phi = (self.uav.iyy - self.uav.izz) / self.uav.ixx * self.d_theta * self.d_psi + 1 / self.uav.ixx * tau_x
        self.dd_theta = (self.uav.izz - self.uav.ixx) / self.uav.iyy * self.d_phi * self.d_psi + 1 / self.uav.iyy * tau_y
        self.dd_psi = (self.uav.ixx - self.uav.iyy) / self.uav.izz * self.d_phi * self.d_theta + 1 / self.uav.izz * tau_z

        self.d_phi += self.dd_phi * dt
        self.d_theta += self.dd_theta * dt
        self.d_psi += self.dd_psi * dt

        self.phi += self.d_phi * dt
        self.theta += self.d_theta * dt
        self.psi += self.d_psi * dt

    def get_position(self):
        return [self.x, self.y, self.z]

    def get_vel(self):
        return [self.d_x, self.d_y, self.d_z]

    def get_acc(self):
        return [self.dd_x, self.dd_y, self.dd_z]

    def get_pose(self):
        return [self.phi, self.theta, self.psi]

    def __str__(self):
        return f'pose: [{self.phi}, {self.theta}, {self.psi}]\n' \
               f'position: [{self.x}, {self.y}, {self.z}]\n' \
               f'vel: [{self.d_x}, {self.d_y}, {self.d_z}]\n' \
               f'acc: [{self.dd_x}, {self.dd_y}, {self.dd_z}]'


def main():
    uav = UavModel()
    uav_dynamics = UavDynamics(uav)

    for i in range(100):
        uav_dynamics.update_with_control([1600., 1600., 1600., 1600.], 1 / 1000)
    print(uav_dynamics)


if __name__ == '__main__':
    main()
