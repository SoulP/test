package cudaTest;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class ITb {
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
            System.out.println("デバイス名: " + p.getName());
            long totalMemKBytes = p.totalGlobalMem / 1024;
            long totalMemMBytes = totalMemKBytes / 1024;
            System.out.println("グローバルメモリの合計: " + totalMemMBytes + " MB");
            long sharedMemPerBlockKB = p.sharedMemPerBlock / 1024;
            System.out.println("1ブロックあたりの共有メモリの合計: " + sharedMemPerBlockKB + " KB");
            System.out.println("1ブロックあたりの使用可能な32ビット命令セット数（レジスタ数）: " + p.regsPerBlock);
            System.out.println("Warp(並列実行の最小単位): " + p.warpSize + " スレッド");
            long memPitchKB = p.memPitch / 1024;
            long memPitchMB = memPitchKB / 1024;
            System.out.println("2次元配列確保される最大メモリ（メモリピッチ）: " + memPitchMB + " MB");
            System.out.println("1ブロックあたりの最大スレッド数: " + p.maxThreadsPerBlock);
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
            System.out.println("オーバーラップ機能: " + (p.deviceOverlap == 1 ? "有" : "無"));
            System.out.println("ストリーミング・マルチプロセッサ数（SM）: " + p.multiProcessorCount);
            System.out.println("カーネル実行タイムアウト機能: " + (p.kernelExecTimeoutEnabled == 1 ? "有" : "無"));
            System.out.println("統合・分割: " + (p.integrated == 1 ? "統合" : "分割"));
            System.out.println(p.toFormattedString());
            System.out.println();
        });
    }
}
