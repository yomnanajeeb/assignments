import java.util.Scanner;

public class question1 {
    public static void main(String[] args){

        Scanner scanner =new Scanner(System.in);
        System.out.println("please enter number");
        int x =scanner.nextInt();
        if (x%2==0)
            System.out.println("number is even");
        else
            System.out.println("number is odd");


    }
}