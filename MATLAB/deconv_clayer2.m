classdef deconv_clayer2 < nnet.layer.Layer

  properties
    idim, odim, hrfd, hsir;
  end

  properties (Learnable)
    hrfp;
  end

  methods
    function layer = deconv_clayer2(hrfd_,idim_,odim_)
      layer.idim = idim_;
      layer.odim = odim_;

      layer.hrfd = hrfd_;
      layer.hrfp = zeros(1,prod(hrfd_)-1)
      %layer.hrf = zeros(hrfd_);
      %layer.hrf(1) = 1;

      [X1,X2] = ndgrid(0:hrfd_(1)-1,0:hrfd_(2)-1);
      hsirf = [X1(:)'; X2(:)'];
      layer.hsir = hsirf(:,2:end);
    end

    function layer = initialize(layer,layout)
      %keyboard;  % No idea when this gets called...
    end
        
    function Z = predict(layer,X)
      xdim = size(X);
      hm1 = zeros(xdim(1), xdim(2));
      hrf = zeros(layer.hrfd);  hrf(1) = 1;  hrf(2:end) = layer.hrfp;
      hm1(1:layer.hrfd(1),1:layer.hrfd(2)) = hrf;
      
      gm1f = 1./fft2(hm1);
      gm2f = circshift(flip(gm1f,1),1,1);
      gm3f = circshift(flip(gm1f,2),1,2);
      gm4f = circshift(flip(gm3f,1),1,1);
      gmf = gm1f.*gm2f.*gm3f.*gm4f;

      ymf = gmf.*fft2(X);
      Z = ifft2(ymf);
    end

    function [Z,memory] = forward(layer,X)
      xdim = size(X);
      hm1 = zeros(xdim(1), xdim(2));
      hrf = zeros(layer.hrfd);  hrf(1) = 1;  hrf(2:end) = layer.hrfp;
      hm1(1:layer.hrfd(1),1:layer.hrfd(2)) = hrf;
      
      gm1f = 1./fft2(hm1);
      gm2f = circshift(flip(gm1f,1),1,1);
      gm3f = circshift(flip(gm1f,2),1,2);
      gm4f = circshift(flip(gm3f,1),1,1);
      gmf = gm1f.*gm2f.*gm3f.*gm4f;

      ymf = gmf.*fft2(X);
      Z = ifft2(ymf);
      memory(:,:,1) = gm1f;  
      memory(:,:,2) = gm2f;
      memory(:,:,3) = gm3f;
      memory(:,:,4) = gm4f;
      memory(:,:,5) = gmf;
      memory(:,:,6) = ymf;
    end

    function [dLdX,dLdHrf] = backward(layer,X,Z,dLdZ,memory)
      um = dLdZ;
      gm1f = memory(:,:,1);
      gm2f = memory(:,:,2);
      gm3f = memory(:,:,3);
      gm4f = memory(:,:,4);
      gmf =  memory(:,:,5);
      ymf =  memory(:,:,6);
      
      dLdX = ifft2(gmf.*fft2(dLdZ));
      cl = class(dLdZ);
      uyphmr = zeros(1,size(layer.hsir,2), cl);
      M = length(size(X));

      % g1
      vm = ifft2(gm1f.*ymf);
      hsirf = layer.hsir;
      for j=1:size(hsirf,2)
        vmq = circshift(vm,hsirf(:,j)');
        uyphmr(j) = uyphmr(j) + dot(um(:),vmq(:));
      end
      
      % g2
      vm = ifft2(gm2f.*ymf);
      hsirf = layer.hsir;  hsirf(1,:) = -hsirf(1,:);
      for j=1:size(hsirf,2)
        vmq = circshift(vm,hsirf(:,j)');
        uyphmr(j) = uyphmr(j) + dot(um(:),vmq(:));
      end

      % g3
      vm = ifft2(gm3f.*ymf);
      hsirf = layer.hsir;  hsirf(2,:) = -hsirf(2,:);
      for j=1:size(hsirf,2)
        vmq = circshift(vm,hsirf(:,j)');
        uyphmr(j) = uyphmr(j) + dot(um(:),vmq(:));
      end

      % g4
      vm = ifft2(gm4f.*ymf);
      hsirf = -layer.hsir;
      for j=1:size(hsirf,2)
        vmq = circshift(vm,hsirf(:,j)');
        uyphmr(j) = uyphmr(j) + dot(um(:),vmq(:));
      end
        
      %uyphm = reshape([0 -uyphmr],layer.hrfd);
      uyphm = -uyphmr;
      dLdHrf = uyphm;

      % Any of the below should make checklayer() fail
      %dLdX = dLdX + 0.001*randn(size(dLdX));
      %dLdW = dLdW + 0.001*randn(size(dLdW));
      %dLdb = dLdb + 0.001*randn(size(dLdb));
    end
  end
end