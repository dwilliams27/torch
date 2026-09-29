// The dream atlas: a persistent per-surface texture that diffusion results are
// projected into. Stored premultiplied: rgb = sum(w_i * c_i) (EMA), a = 1 - prod(1 - w_i)
// (confidence), so rgb/a is a clean weighted average and unpainted padding
// texels never darken bilinear/mip samples (seam-free without dilation).
import * as THREE from 'three';

export class Painter {
  constructor(renderer, level, chunkGeos, materials, { atlasSize, halfFloat = true } = {}) {
    this.renderer = renderer;
    this.materials = materials;
    const size = atlasSize || level.atlas?.size || 4096;
    this.size = size;
    this.atlas = new THREE.WebGLRenderTarget(size, size, {
      type: halfFloat ? THREE.HalfFloatType : THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      generateMipmaps: true,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      anisotropy: Math.min(8, renderer.capabilities.getMaxAnisotropy()),
      colorSpace: THREE.NoColorSpace,
    });
    this.atlas.texture.generateMipmaps = true;
    // paint scene: the same chunk geometries, but with the paint material
    this.scene = new THREE.Scene();
    this.scene.matrixWorldAutoUpdate = false;
    this.meshes = [];
    for (const g of chunkGeos) {
      const m = new THREE.Mesh(g, materials.paint);
      m.matrixAutoUpdate = false;
      m.frustumCulled = false; // culled manually (frustum + paint-relevance distance)
      m.updateMatrixWorld(true);
      m.userData.far = g.boundingSphere.radius > 60; // sky/backdrop buckets: always relevant
      this.scene.add(m);
      this.meshes.push(m);
    }
    this.frustum = new THREE.Frustum();
    this.maxPaintDist = 140; // beyond this the distance weight is ~0; skip rasterizing
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));
    this.quad = new THREE.Mesh(geo, materials.relax);
    this.quad.frustumCulled = false;
    this.quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.paints = 0;
    this.clear();
  }

  get texture() { return this.atlas.texture; }

  clear() {
    const r = this.renderer;
    const prev = r.getRenderTarget();
    const cc = r.getClearColor(new THREE.Color()), ca = r.getClearAlpha();
    r.setRenderTarget(this.atlas);
    r.setClearColor(0x000000, 0);
    r.clear(true, false, false);
    r.setClearColor(cc, ca);
    r.setRenderTarget(prev);
  }

  // job: { slot (capture record), texture, flipY, rate }
  paint(job) {
    const r = this.renderer;
    const U = this.materials.paint.uniforms;
    const s = job.slot;
    U.uImage.value = job.texture;
    U.uFlipY.value = job.flipY ? 1 : 0;
    U.uDepth.value = s.rt.depthTexture;
    U.uCapViewProj.value.copy(s.viewProj);
    U.uCapPos.value.copy(s.pos);
    U.uNear.value = s.near; U.uFar.value = s.far;
    U.uRate.value = job.rate;
    U.uSide.value = s.side || 0;
    U.uDepthTexel.value.set(1 / s.rt.width, 1 / s.rt.height);
    U.uPixelAngle.value = 2 * Math.tan(THREE.MathUtils.degToRad(s.fov) / 2) / s.rt.height;
    this.frustum.setFromProjectionMatrix(s.viewProj);
    let n = 0;
    for (const m of this.meshes) {
      const sp = m.geometry.boundingSphere;
      m.visible = this.frustum.intersectsSphere(sp) && (m.userData.far || sp.center.distanceTo(s.pos) - sp.radius < this.maxPaintDist);
      n += m.visible ? 1 : 0;
    }
    this.lastDraws = n;
    const prev = r.getRenderTarget();
    r.setRenderTarget(this.atlas);
    r.render(this.scene, s.camera);
    r.setRenderTarget(prev);
    this.paints++;
  }

  relax(factor) {
    const r = this.renderer;
    this.materials.relax.uniforms.uFactor.value = factor;
    this.quad.material = this.materials.relax;
    const prev = r.getRenderTarget();
    r.setRenderTarget(this.atlas);
    r.render(this.quad, this.quadCam);
    r.setRenderTarget(prev);
  }
}
