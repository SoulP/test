package cudaTest;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class UT {
    @Test
    public void test001() {
        int[] count = new int[] {-1};
        JCuda.cudaGetDeviceCount(count);
        System.out.println("デバイス数: " + count[0]);
        System.out.println();

        List<cudaDeviceProp> propList = new ArrayList<>();

        for(int i = 0; i < count[0]; i++) {
            cudaDeviceProp prop = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(prop, i);
            propList.add(prop);
        }

        propList.forEach(p -> {
            System.out.println("デバイス名: " + p.getName());
            long totalMemKBytes = p.totalGlobalMem / 1024;
            long totalMemMBytes = totalMemKBytes / 1024;
            System.out.println("合計メモリ: " + totalMemMBytes + " MB");
            long sharedMemPerBlockKB = p.sharedMemPerBlock / 1024;
            System.out.println("1ブロック共有される最大メモリ: " + sharedMemPerBlockKB + " KB");
            System.out.println();
            int clockMHz = p.clockRate / 1000;
            System.out.println("クロック周波数: " + clockMHz + " MHz");
            System.out.println(p.toFormattedString());
            System.out.println();
        });
    }
}
