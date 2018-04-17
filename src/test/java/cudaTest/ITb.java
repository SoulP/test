package cudaTest;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class ITb {

    /**
     * 各デバイスプロパティ取得
     */
    @Test
    public void test001() {
        int[] count = new int[] { -1 };
        JCuda.cudaGetDeviceCount(count);
        System.out.println("デバイス数: " + count[0]);
        System.out.println();

        List<cudaDeviceProp> propList = new ArrayList<>();

        for (int i = 0; i < count[0]; i++) {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, i);
            propList.add(prop);
        }

        propList.forEach(p -> {
            System.out.println("デバイスプロパティ（日本語）");
            System.out.println("デバイス名: " + p.getName());
            long totalMemKBytes = p.totalGlobalMem / 1024;
            long totalMemMBytes = totalMemKBytes / 1024;
            System.out.println("グローバルメモリの合計: " + totalMemMBytes + " MB");
            long sharedMemPerBlockKB = p.sharedMemPerBlock / 1024;
            System.out.println("ブロックあたりの共有メモリ: " + sharedMemPerBlockKB + " KB");
            System.out.println("ブロックあたりの使用可能な32bitレジスタ数: " + p.regsPerBlock);
            System.out.println("Warp(並列実行の最小単位): " + p.warpSize + " スレッド");
            long memPitchKB = p.memPitch / 1024;
            long memPitchMB = memPitchKB / 1024;
            System.out.println("2次元配列確保される最大メモリ（メモリピッチ）: " + memPitchMB + " MB");
            System.out.println("ブロックあたりの最大スレッド数: " + p.maxThreadsPerBlock);
            System.out.printf("ブロックの各次元の最大サイズ: %d, %d, %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1],
                    p.maxThreadsDim[2]);
            System.out.printf("グリッドの各次元の最大サイズ: %d, %d, %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
            int clockMHz = p.clockRate / 1000;
            System.out.println("クロック周波数: " + clockMHz + " MHz");
            long totalConstMemKB = p.totalConstMem / 1024;
            System.out.println("コンスタントメモリの合計: " + totalConstMemKB + " KB");
            System.out.println("バージョン: " + p.major + "." + p.minor);
            System.out.println("テクスチャアライメント: " + p.textureAlignment);
            System.out.println("テクスチャピッチアライメント: " + p.texturePitchAlignment);
            System.out.println("オーバーラップ機能: " + (p.deviceOverlap != 0 ? "有" : "無"));
            System.out.println("ストリーミング・マルチプロセッサ数（SM）: " + p.multiProcessorCount);
            System.out.println("カーネル実行タイムアウト機能: " + (p.kernelExecTimeoutEnabled != 0 ? "有" : "無"));
            System.out.println("統合・分割: " + (p.integrated != 0 ? "統合" : "分割"));
            System.out.println("デバイスのアドレス空間にPage-lockedホストメモリブロックをマップの可否: " + (p.canMapHostMemory != 0 ? "可" : "不可"));
            String mode;
            switch (p.computeMode) {
                case 0:
                    mode = "cudaComputeModeDefault (複数スレッド使用可)";
                    break;
                case 1:
                    mode = "cudaComputeModeExclusive (スレッド1つだけ使用可)";
                    break;
                case 2:
                    mode = "cudaComputeModeProhibited (スレッド使用不可)";
                    break;
                default:
                    mode = "不明";
            }
            System.out.println("コンピュート・モード: " + mode);
            System.out.println("1次元テクスチャ最大サイズ: " + p.maxTexture1D);
            System.out.println("ミップマップによる1次元テクスチャ最大サイズ: " + p.maxTexture1DMipmap);
            System.out.println("1次元テクスチャ配列の最大サイズ: " + p.maxTexture1DLinear);
            System.out.println("2次元テクスチャ最大サイズ: 幅 " + p.maxTexture2D[0] + ", 高さ " + p.maxTexture2D[1]);
            System.out.println(
                    "ミップマップによる2次元テクスチャ最大サイズ: 幅 " + p.maxTexture2DMipmap[0] + ", 高さ " + p.maxTexture2DMipmap[1]);
            System.out.println("2次元テクスチャ配列の最大サイズ: 幅 " + p.maxTexture2DLinear[0] + ", 高さ " + p.maxTexture2DLinear[1]
                    + ", ピッチ " + p.maxTexture2DLinear[2]);
            System.out
                    .println("ギャザーによる2次元テクスチャの最大サイズ: 幅 " + p.maxTexture2DGather[0] + ", 高さ " + p.maxTexture2DGather[1]);
            System.out.println("3次元テクスチャ最大サイズ: 幅 " + p.maxTexture3D[0] + ", 高さ " + p.maxTexture3D[1] + ", 奥行き "
                    + p.maxTexture3D[2]);
            System.out.println("代替の3次元テクスチャ最大サイズ: 幅 " + p.maxTexture3DAlt[0] + ", 高さ " + p.maxTexture3DAlt[1] + ", 奥行き "
                    + p.maxTexture3DAlt[2]);
            System.out.println("キューブマップによるテクスチャの最大サイズ: " + p.maxTextureCubemap);
            System.out.println("1次元レイヤテクスチャ最大サイズ: " + p.maxTexture1DLayered[0] + ", " + p.maxTexture1DLayered[1]);
            System.out.println("2次元レイヤテクスチャ最大サイズ: " + p.maxTexture2DLayered[0] + ", " + p.maxTexture2DLayered[1] + ", "
                    + p.maxTexture2DLayered[2]);
            System.out.println("1次元サーフェス最大サイズ: " + p.maxSurface1D);
            System.out.println("2次元サーフェス最大サイズ: 幅 " + p.maxSurface2D[0] + ", 高さ " + p.maxSurface2D[1]);
            System.out.println("3次元サーフェス最大サイズ: 幅 " + p.maxSurface3D[0] + ", 高さ " + p.maxSurface3D[1] + ", 奥行き "
                    + p.maxSurface3D[2]);
            System.out.println("1次元レイヤサーフェス最大サイズ: " + p.maxSurface1DLayered[0] + ", " + p.maxSurface1DLayered[1]);
            System.out.println("2次元レイヤサーフェス最大サイズ: " + p.maxSurface2DLayered[0] + ", " + p.maxSurface2DLayered[1] + ", "
                    + p.maxSurface2DLayered[2]);
            System.out.println("キューブマップによるサーフェス最大サイズ: " + p.maxSurfaceCubemap);
            System.out.println(
                    "キューブマップによるレイヤサーフェス最大サイズ: " + p.maxSurfaceCubemapLayered[0] + ", " + p.maxSurfaceCubemapLayered[1]);
            System.out.println("サーフェスに必要なアライメント: " + p.surfaceAlignment);
            System.out.println("並行カーネル機能: " + (p.concurrentKernels != 0 ? "有" : "無"));
            System.out.println("ECC機能: " + (p.ECCEnabled != 0 ? "有" : "無"));
            System.out.println("PCI バス ID: " + p.pciBusID);
            System.out.println("PCI デバイス ID: " + p.pciDeviceID);
            System.out.println("PCI ドメイン ID: " + p.pciDomainID);
            System.out.println("ドライバ・モード: " + (p.tccDriver != 0 ? "TCC" : "WDDM"));
            System.out.println("同期エンジン: " + p.asyncEngineCount + "本");
            System.out.println("UVA(Unified Virtual Addressing)機能: " + (p.unifiedAddressing != 0 ? "有" : "無"));
            long memoryClockRateMHz = p.memoryClockRate / 1000;
            System.out.println("メモリクロック周波数: " + memoryClockRateMHz + " MHz");
            System.out.println("メモリバス幅: " + p.memoryBusWidth + " bit");
            long l2CacheSizeKB = p.l2CacheSize / 1024;
            System.out.println("L2キャッシュサイズ: " + l2CacheSizeKB + " KB");
            System.out.println("マルチプロセッサあたりの最大スレッド数: " + p.maxThreadsPerMultiProcessor);
            System.out.println("ストリームの優先順位対応: " + (p.streamPrioritiesSupported != 0 ? "有" : "無"));
            System.out.println("グローバルL1キャッシュ対応: " + (p.globalL1CacheSupported != 0 ? "有" : "無"));
            System.out.println("ローカルL1キャッシュ対応: " + (p.localL1CacheSupported != 0 ? "有" : "無"));
            long sharedMemPerMultiprocessorKB = p.sharedMemPerMultiprocessor / 1024;
            System.out.println("マルチプロセッサあたりの共有メモリ: " + sharedMemPerMultiprocessorKB + " KB");
            System.out.println("マルチプロセッサあたりの32bitレジスタ数" + p.regsPerMultiprocessor);
            System.out.println("メモリ管理: " + (p.managedMemory != 0 ? "有" : "無"));
            System.out.println("マルチGPUボードなのか: " + (p.isMultiGpuBoard != 0 ? "はい" : "いいえ"));
            System.out.println("マルチGPUボードグループID: " + p.multiGpuBoardGroupID);
            System.out.println("ホストネイティブアトミック対応: " + (p.hostNativeAtomicSupported != 0 ? "有" : "無"));
            System.out.println("単精度から倍精度のパフォーマンス比率: " + p.singleToDoublePrecisionPerfRatio);
            System.out.println("pageable メモリアクセス: " + (p.pageableMemoryAccess != 0 ? "可" : "不可"));
            System.out.println("並行管理対象アクセス: " + (p.concurrentManagedAccess != 0 ? "可" : "不可"));
            System.out.println("計算プリエンプション対応: " + (p.computePreemptionSupported != 0 ? "有" : "無"));
            System.out.println(
                    "登録されたメモリのホストポインタを使用することが: " + (p.canUseHostPointerForRegisteredMem != 0 ? "できる" : "できない"));
            System.out.println("協調起動: " + (p.cooperativeLaunch != 0 ? "可" : "不可"));
            System.out.println("複数デバイス協調起動: " + (p.cooperativeMultiDeviceLaunch != 0 ? "可" : "不可"));
            System.out.println("ブロックあたりの共有メモリ（オプション）: " + p.sharedMemPerBlockOptin);
            System.out.println();
            System.out.println("デバイスプロパティ（英語）");
            System.out.println(p.toFormattedString());
            System.out.println();
            System.out.println();
        });
    }
}
