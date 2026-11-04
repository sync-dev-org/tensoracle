#pragma once
#include <tensoracle/tensoracle.hpp>
#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace tensoracle::detail {
namespace native_detail {
[[noreturn]] inline void invalid(const std::string& message) {
    throw tensoracle::error(TENSORACLE_INVALID_ARGUMENT,message.c_str());
}
inline void require(bool value,const std::string& message) { if(!value) invalid(message); }
inline uint64_t number(const std::string& bytes,uint64_t offset,size_t count) {
    require(offset<=bytes.size() && count<=bytes.size()-offset,"truncated native library");
    uint64_t result=0;
    for(size_t i=0;i<count;++i) result|=uint64_t(static_cast<unsigned char>(bytes[static_cast<size_t>(offset)+i]))<<(i*8);
    return result;
}
struct Symbol { uint64_t offset; uint64_t size; bool exact_size; };
inline std::map<std::string,Symbol> symbols(const std::string& data) {
    auto n=[&](uint64_t offset,size_t size){return number(data,offset,size);};
    std::map<std::string,Symbol> result;
    if(data.compare(0,2,"MZ")==0) {
        auto header=n(0x3c,4);
        require(header<=data.size() && data.compare(static_cast<size_t>(header),4,"PE\0\0",4)==0,"invalid PE plugin library");
        auto sections=n(header+6,2),optional=header+24;
        auto magic=n(optional,2);
        require(magic==0x20b || magic==0x10b,"unsupported PE plugin library");
        auto section_table=optional+n(header+20,2);
        require(section_table<=data.size() && sections<=(data.size()-section_table)/40,"truncated PE section table");
        auto locate=[&](uint64_t rva,size_t bytes) -> Symbol {
            for(uint64_t i=0;i<sections;++i) {
                auto s=section_table+i*40,address=n(s+12,4),raw=n(s+16,4),start=n(s+20,4);
                if(rva>=address && rva-address<raw && bytes<=raw-(rva-address)) {
                    auto offset=start+rva-address,available=raw-(rva-address);
                    require(offset<=data.size() && available<=data.size()-offset,"truncated PE export data");
                    return {offset,available,false};
                }
            }
            invalid("invalid PE export address");
        };
        auto offset=[&](uint64_t rva,size_t size){return locate(rva,size).offset;};
        auto export_rva=n(optional+(magic==0x20b?112:96),4);
        if(export_rva==0) return result;
        auto exports=offset(export_rva,40),count=n(exports+24,4);
        require(count<=100000,"invalid PE export count");
        auto names=n(exports+32,4),ordinals=n(exports+36,4),functions=n(exports+28,4),function_count=n(exports+20,4);
        auto export_size=n(optional+(magic==0x20b?116:100),4);
        for(uint64_t i=0;i<count;++i) {
            auto name=offset(n(offset(names+i*4,4),4),1),end=data.find('\0',static_cast<size_t>(name));
            require(end!=std::string::npos,"unterminated PE export name");
            auto ordinal=n(offset(ordinals+i*2,2),2);
            require(ordinal<function_count,"invalid PE export ordinal");
            auto rva=n(offset(functions+ordinal*4,4),4);
            // Forwarded exports name another DLL, not bytes in this file.
            if(rva==0 || (rva>=export_rva && rva-export_rva<export_size)) continue;
            result.emplace(data.substr(static_cast<size_t>(name),end-name),locate(rva,1));
        }
        return result;
    }
    require(data.size()>=64 && data.compare(0,6,"\177ELF\2\1",6)==0,"expected 64-bit little-endian ELF plugin library");
    auto table=n(40,8),stride=n(58,2),count=n(60,2);
    require(stride>=64 && count>0 && table<=data.size() && count<=(data.size()-table)/stride,"invalid ELF section table");
    for(uint64_t i=0;i<count;++i) {
        auto section=table+i*stride;
        if(n(section+4,4)!=11) continue;
        auto string_index=n(section+40,4);
        require(string_index<count,"invalid ELF string table");
        auto strings=table+string_index*stride,names=n(strings+24,8),names_size=n(strings+32,8);
        require(names<=data.size() && names_size<=data.size()-names,"truncated ELF string table");
        auto symbols=n(section+24,8),bytes=n(section+32,8),step=n(section+56,8);
        require(step>=24 && bytes%step==0 && symbols<=data.size() && bytes<=data.size()-symbols,"invalid ELF symbol table");
        for(uint64_t j=0;j<bytes;j+=step) {
            auto symbol=symbols+j,ni=n(symbol,4),info=n(symbol+4,1),visibility=n(symbol+5,1)&3,owner_index=n(symbol+6,2);
            if(owner_index==0 || (info>>4!=1 && info>>4!=2) || (visibility!=0 && visibility!=3)) continue;
            require(ni<names_size,"invalid ELF symbol name");
            auto end=data.find('\0',static_cast<size_t>(names+ni));
            require(end!=std::string::npos && end<names+names_size,"unterminated ELF symbol name");
            auto name=data.substr(static_cast<size_t>(names+ni),end-names-ni);
            if(owner_index>=count) { result.emplace(name,Symbol{0,0,true}); continue; }
            auto owner=table+owner_index*stride,address=n(symbol+8,8),base=n(owner+16,8),start=n(owner+24,8);
            auto owner_size=n(owner+32,8),symbol_size=n(symbol+16,8);
            if(n(owner+4,4)==8 || address<base || address-base>owner_size || symbol_size>owner_size-(address-base)) {
                result.emplace(name,Symbol{0,0,true}); continue;
            }
            require(start<=data.size() && address-base<=data.size()-start && symbol_size<=data.size()-start-(address-base),"invalid ELF symbol data");
            result.emplace(name,Symbol{start+address-base,symbol_size,true});
        }
    }
    return result;
}
}
/* These inspect file bytes only. No dlopen/LoadLibrary, CUDA, or constructors. */
inline std::set<std::string> native_exports(const std::string& bytes) {
    std::set<std::string> result;
    for(auto const& [name,symbol]:native_detail::symbols(bytes)) result.insert(name);
    return result;
}
/* ELF carries the exact symbol size. PE requires expected_size > 0 because its
 * export directory carries addresses but no constant sizes. */
inline std::string exported_data(const std::string& bytes,const std::string& name,size_t expected_size=0) {
    auto symbols=native_detail::symbols(bytes);
    auto found=symbols.find(name);
    native_detail::require(found!=symbols.end(),"native library does not export "+name);
    auto symbol=found->second;
    native_detail::require(symbol.exact_size || expected_size>0,"PE data inspection requires a positive size");
    auto size=expected_size?expected_size:symbol.size;
    native_detail::require(size>0 && (symbol.exact_size?symbol.size==size:symbol.size>=size),"native data symbol "+name+" has the wrong size");
    return bytes.substr(static_cast<size_t>(symbol.offset),size);
}
}
