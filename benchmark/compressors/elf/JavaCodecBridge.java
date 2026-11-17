// Thin byte-stream bridge to the unmodified upstream Java codec classes.
import java.io.*;
import java.lang.reflect.*;
import java.nio.*;
import java.util.*;

public class JavaCodecBridge {
    static String encoder(String algorithm,int width) {
        if(algorithm.equals("chimp"))return "gr.aueb.delorean.chimp.Chimp"+(width==4?"32":"");
        return "org.urbcomp.startdb.compress.elf.compressor"+(width==4?"32":"")+".ElfCompressor"+(width==4?"32":"");
    }
    static String decoder(String algorithm,int width) {
        if(algorithm.equals("chimp"))return "gr.aueb.delorean.chimp.ChimpDecompressor"+(width==4?"32":"");
        return "org.urbcomp.startdb.compress.elf.decompressor"+(width==4?"32":"")+".ElfDecompressor"+(width==4?"32":"");
    }
    static byte[] decodeFrame(String algorithm,int width,byte[] encoded,int count)throws Exception {
        Class<?> type=Class.forName(decoder(algorithm,width));
        Object object=type.getConstructor(byte[].class).newInstance((Object)encoded);
        List<?> values;
        if(algorithm.equals("chimp")&&width==4){
            List<Float> floats=new ArrayList<>();Method next=type.getMethod("readPair");Object value;
            while((value=next.invoke(object))!=null){floats.add((Float)value.getClass().getMethod("getFloatValue").invoke(value));if(floats.size()>count)throw new IOException("Too many decoded values");}
            values=floats;
        }else values=(List<?>)type.getMethod(algorithm.equals("chimp")?"getValues":"decompress").invoke(object);
        if(values.size()!=count)throw new IOException("Upstream element count mismatch");
        ByteBuffer out=ByteBuffer.allocate(count*width).order(ByteOrder.LITTLE_ENDIAN);
        for(Object value:values){if(width==4)out.putInt(Float.floatToRawIntBits(((Number)value).floatValue()));else out.putLong(Double.doubleToRawLongBits(((Number)value).doubleValue()));}
        return out.array();
    }
    static byte[] encodeFrame(String algorithm,int width,byte[] raw)throws Exception {
        ByteBuffer input=ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);int n=raw.length/width;
        Class<?> type=Class.forName(encoder(algorithm,width));Object object=type.getConstructor().newInstance();
        Method add=type.getMethod("addValue",width==4?float.class:double.class);
        for(int i=0;i<n;i++){
            if(width==4){float v=Float.intBitsToFloat(input.getInt());if(Float.isNaN(v))return null;add.invoke(object,v);}
            else {double v=Double.longBitsToDouble(input.getLong());if(Double.isNaN(v))return null;add.invoke(object,v);}
        }
        type.getMethod("close").invoke(object);
        byte[] buffer=(byte[])type.getMethod(algorithm.equals("chimp")?"getOut":"getBytes").invoke(object);
        int bits=((Number)type.getMethod("getSize").invoke(object)).intValue();
        byte[] encoded=Arrays.copyOf(buffer,Math.min(buffer.length,(bits+7)/8+32));
        if(!Arrays.equals(raw,decodeFrame(algorithm,width,encoded,n)))return null;
        return encoded;
    }
    public static void main(String[] args)throws Exception {
        String operation=args[0],algorithm=args[1];int width=Integer.parseInt(args[2]);
        if(width!=4&&width!=8)throw new IOException("Invalid width");
        byte[] input=System.in.readAllBytes();ByteArrayOutputStream bytes=new ByteArrayOutputStream();DataOutputStream out=new DataOutputStream(bytes);
        if(operation.equals("encode")){
            if(input.length%width!=0)throw new IOException("Unaligned input");
            out.writeInt(0x4a434231);out.writeInt(width);out.writeInt(input.length);
            for(int offset=0;offset<input.length;offset+=512*width){
                byte[] raw=Arrays.copyOfRange(input,offset,Math.min(input.length,offset+512*width));byte[] encoded=null;
                try{encoded=encodeFrame(algorithm,width,raw);}catch(Exception unsupported){encoded=null;}
                boolean compressed=encoded!=null&&encoded.length<raw.length;
                byte[] payload=compressed?encoded:raw;out.writeByte(compressed?1:0);out.writeInt(raw.length);out.writeInt(payload.length);out.write(payload);
            }
        }else if(operation.equals("decode")){
            DataInputStream in=new DataInputStream(new ByteArrayInputStream(input));
            if(in.readInt()!=0x4a434231||in.readInt()!=width)throw new IOException("Invalid bridge header");int expected=in.readInt();
            if(expected<0||expected>64*1024*1024||expected%width!=0)throw new IOException("Invalid output size");
            while(bytes.size()<expected){int mode=in.readUnsignedByte(),rawSize=in.readInt(),size=in.readInt();
                if(rawSize<=0||rawSize>512*width||rawSize%width!=0||size<0||size>65536||rawSize>expected-bytes.size())throw new IOException("Invalid frame size");
                byte[] payload=in.readNBytes(size);if(payload.length!=size)throw new EOFException();
                byte[] raw=mode==0?payload:mode==1?decodeFrame(algorithm,width,payload,rawSize/width):null;
                if(raw==null||raw.length!=rawSize)throw new IOException("Invalid decoded frame");out.write(raw);
            }
            if(in.available()!=0)throw new IOException("Trailing bridge data");
        }else throw new IOException("Invalid operation");
        System.out.write(bytes.toByteArray());
    }
}
