module mod_kernels
    use cudafor
    use openacc

    implicit none

    contains

    subroutine vecAdd(n, a, b, c)
        implicit none
        integer(8), intent(in) :: n
        real(4), intent(in)    :: a(n), b(n)
        real(4), intent(out)   :: c(n)
        integer(8) :: i

        !$acc parallel loop
        do i = 1, n
            c(i) = a(i) + b(i)
        end do
        !$acc end parallel loop
    end subroutine vecAdd

    subroutine matmul_naive(n, A, B, C)
        implicit none
        integer(8), intent(in) :: n
        real(4), intent(in)    :: A(n,n),B(n,n)
        real(4), intent(out)   :: C(n,n)
        integer(8) :: i, j, k

        !$acc kernels
        C(:,:) = 0.0
        !$acc end kernels

        !$acc parallel loop collapse(2)
        do i = 1,n
            do j = 1,n
                !$acc loop seq
                do k = 1,n
                    C(i,j) = C(i,j) + A(i,k) * B(k,j)
                end do
            end do
        end do
        !$acc end parallel loop
    end subroutine matmul_naive

    subroutine fem_laplacian(nelem,ndime,ngaus,nnode,connec,phig,He,dNgp,gpvol,Rg)
        implicit none
        integer(8), intent(in) :: nelem, ndime, nnode, ngaus
        integer(8), intent(in) :: connec(nelem,nnode)
        real(4), intent(in)    :: phig(:), He(ndime,ndime), dNgp(ndime,nnode,ngaus), gpvol(ngaus,nelem)
        real(4), intent(out)   :: Rg(:)
        integer(8)             :: ielem, idime, igaus, inode, ipoin(nnode)
        real(4)                :: phil(nnode), Re(nnode), gpcar(ndime,nnode), aux

        !$acc kernels
        Rg(:) = 0.0
        !$acc end kernels

        !$acc parallel loop gang private(ipoin,phil,Re,gpcar)
        do ielem = 1, nelem
            !! Gather ops
            !$acc loop vector
            do inode = 1,nnode
                ipoin(inode) = connec(ielem,inode)
                Re(inode) = 0.0
            end do
            !$acc loop vector
            do inode = 1,nnode
                phil(inode) = phig(ipoin(inode))
            end do
            !! Gasussian quadrature
            !$acc loop seq private(aux)
            do igaus = 1,ngaus
                !! Compute gpcar
                !$acc loop vector collapse(2)
                do idime = 1,ndime
                    do inode = 1,nnode
                        gpcar(idime,inode) = dot_product(He(idime,:), dNgp(:,inode,igaus))
                    end do
                end do
                !! compute local residual
                !$acc loop seq
                do idime = 1,ndime
                    aux = dot_product(gpcar(idime,:), phil(:))
                    !$acc loop vector
                    do inode = 1,nnode
                        Re(inode) = Re(inode) + gpvol(igaus,ielem) * gpcar(idime,inode) * aux
                    end do
                end do
            end do
            !! Assembly
            !$acc loop vector
            do inode = 1,nnode
                !$acc atomic update
                Rg(ipoin(inode)) = Rg(ipoin(inode)) + Re(inode)
                !$acc end atomic
            end do
        end do
        !$acc end parallel loop
    end subroutine fem_laplacian
end module mod_kernels