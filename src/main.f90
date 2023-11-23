program optimTest

    use cudafor
    use openacc
    use mod_kernels

    implicit none
    integer(4) :: nruns, irun
    integer(8) :: narr, nmat
    integer(8) :: i, j, k
    real(4), allocatable :: avec(:), bvec(:), cvec(:)
    real(4), allocatable :: amat(:,:), bmat(:,:), cmat(:,:)

    ! Base data section
    nruns = 1
    narr = 512**3
    nmat = 1024

    ! vecAdd

    write(*,*) "Array size: ", narr
    write(*,*) "Memory size: ", real(narr,8)*4.0d0*3.0d0/1024.0d0/1024.0d0, " MB"
    write(*,*) "Number of runs: ", nruns


    allocate(avec(narr), bvec(narr), cvec(narr))
    !$acc enter data create(avec, bvec, cvec)

    !$acc kernels present(avec, bvec, cvec)
    do i = 1, narr
        avec(i) = real(i,4)
        bvec(i) = 1.0
        cvec(i) = 0.0
    end do
    !$acc end kernels

    do irun = 1, nruns
        call vecAdd(narr, avec, bvec, cvec)
    end do

    !$acc update host(cvec)
    write(*,*) "min(cvec) = ", minval(cvec), " max(cvec) = ", maxval(cvec)
    write(*,*) ""

    !$acc exit data delete(avec,bvec,cvec)
    deallocate(avec, bvec, cvec)

    ! matMul

    write(*,*) "Matrix size: ", nmat**2
    write(*,*) "Memory size: ", real(nmat**2,8)*4.0d0*3.0d0/1024.0d0/1024.0d0, " MB"

    allocate(amat(nmat,nmat), bmat(nmat,nmat), cmat(nmat,nmat))
    !$acc enter data create(amat, bmat, cmat)

    !$acc kernels present(amat, bmat, cmat)
    do i = 1, nmat
        do j = 1, nmat
            amat(i,j) = real(i+j,4)/1000000.0
            bmat(i,j) = real(i*j,4)/1000000.0
            cmat(i,j) = 0.0
        end do
    end do
    !$acc end kernels

    write(*,*) "Start..."
    do irun = 1, nruns
        call matmul_naive(nmat, amat, bmat, cmat)
    end do
    write(*,*) "Done!"

    !$acc update host(cmat)
    write(*,*) "cmat(1,1) = ", cmat(1,1), " cmat(1,nmat) = ", cmat(1,nmat)
    write(*,*) "cmat(nmat,1) = ", cmat(nmat,1), " cmat(nmat,nmat) = ", cmat(nmat,nmat)
    write(*,*) ""

end program optimTest